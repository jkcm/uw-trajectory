#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri July  20 14:17:57 2018

@author: jkcm
"""

from . import utils
from . import met_utils
# from CSET_LES import utils as les_utils
from . import les_utils
import datetime as dt
import numpy as np
import os
import xarray as xr
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = 12,5
import matplotlib.pyplot as plt
import glob
import pandas as pd
from itertools import cycle
from geographiclib.geodesic import Geodesic
import time


def rounder(values):
    def f(x):
        idx = np.argmin(np.abs(values - x))
        return values[idx]
    return np.frompyfunc(f, 1, 1)


def xarray_from_trajectory(rfnum, trajnum, trajectory_type='500m_+72'):
    tdump = utils.load_flight_trajectory(rfnum, trajnum, trajectory_type=trajectory_type)
    ds = xr.Dataset.from_dataframe(tdump).drop(['tnum', 'gnum', 'age'])
    ds = ds.rename({'dtime': 'time'})

    # assigning global attributes
    global_attrs = [
        {'Title': "CSET Unified Trajectory Product"},
        {'institution': "Department of Atmospheric Sciences, University of Washington"},
        {'contact': "jkcm@uw.edu"},
        {'trajectory_setup': "Trajectories were run isobarically " +
                            "from an initialization height of 500m " +
                            "for 72 hours, using GDAS analysis met data"},
        {'HYSPLIT_params': "Trajectories run using HYSPLIT (Hybrid Single "+
                   "Particle Lagrangian Integrated Trajectory Model). "+
                   "Acknowledgements to the NOAA Air Resources Laboratory "+
                   "(ARL) for the provision of the HYSPLIT transport and "+
                   "dispersion model used in this publication."},
        {'HYSPLIT_reference': "Stein, A.F., Draxler, R.R, Rolph, G.D., Stunder, "+
                      "B.J.B., Cohen, M.D., and Ngan, F., (2015). NOAA's "+
                      "HYSPLIT atmospheric transport and dispersion modeling "+ 
                      "system, Bull. Amer. Meteor. Soc., 96, 2059-2077, "+
                      "http://dx.doi.org/10.1175/BAMS-D-14-00110.1"},
        {'CSET_flight': rfnum},
        {'flight_trajectory': str(trajnum)}]
    for i in global_attrs:  # note: an OrderedDict would be tidier, but does not unpack in order
        ds = ds.assign_attrs(**i)
    
    # assigning variable attributes
    var_attrs = {
        'lon': {'long_name': 'longitude', 
                'units': 'degrees N'},
        'lat': {'long_name': 'latitude',
                'units': 'degrees E'},
        'fhour': {'long_name': 'forecast_lead_time',
                  'units': 'hours'},
        'pres': {'long_name':'trajectory_pressure',
                 'units': 'hPa'},
        'height': {'long_name': 'trajectory_height_above_ground',
                  'units': 'meters'}}
    for k,v in var_attrs.items():
        ds[k] = ds[k].assign_attrs(**v)
    ds.time.attrs['long_name'] = 'time'
    return ds


def save_trajectory_to_netcdf(ds, location):
    ds.to_netcdf(location)


def add_ERA_ens_to_trajectory(ds, box_degrees=2):
    
    lats, lons, times = ds.lat.values, ds.lon.values, ds.time.values
    space_index = int(np.round(box_degrees/0.3/2)) # go up/down/left/right this many pixels
    ens_files = [os.path.join(utils.ERA_ens_source, i) for i in sorted(os.listdir(utils.ERA_ens_source))]
    

    with xr.open_mfdataset(sorted(ens_files), combine='by_coords') as data:
        data = data.rename({'level': 'ens_level'})
        ds.coords['number'] = data.coords['number']
        ds.coords['ens_level'] = data.coords['ens_level']
        
                    
        if 'w' in data.data_vars.keys() and 'sp' in data.data_vars.keys():
            data['dspdt'] = (data.sp.dims, np.gradient(data.sp, np.median(np.gradient(data.time.values)/np.timedelta64(1, 's')), axis=0),
                    {'units': "Pa s**-1", 'long_name': "Surface pressure tendency", 'standard_name': 'tendency_of_surface_air_pressure'})
            data['w_corr'] = (data.w.dims, data.w - data.dspdt, {'units': data.w.units, 'long_name': 'Vertical velocity (sp-corrected)'})
        
        
        for var in data.data_vars.keys():
            var_shape = data[var].isel(time=0, latitude=0, longitude=0).shape
            vals = []
            for (lat, lon, time) in zip(lats, lons%360, times):
                if lat > np.max(data.coords['latitude']) or lat < np.min(data.coords['latitude']) or \
                    lon > np.max(data.coords['longitude']) or lon < np.min(data.coords['longitude']):
                    print(f'out of range of data" {lat}, {lon}, {time}')
                    vals.append(np.full(var_shape, float('nan'), dtype='float'))
                    continue
                x = data[var].sel(longitude=slice(lon - box_degrees/2, lon + box_degrees/2),
                                  latitude=slice(lat + box_degrees/2, lat - box_degrees/2))
                z = x.sel(method='nearest', time=time, tolerance=np.timedelta64(1, 'h'))
                #this applies a 2D gaussian the width of z, i.e. sigma=box_degrees
                gauss_shape = tuple([v for v,i in zip(z.shape,z.dims) if i in ['latitude', 'longitude'] ])
                gauss = utils.gauss2D(shape=gauss_shape, sigma=gauss_shape[-1])
                filtered = z * gauss
                vals.append(filtered.sum(dim=('latitude', 'longitude')).values)
            ds['ERA_ens_'+var] = (tuple(x for x in data[var].dims if x not in ['latitude', 'longitude']), np.array(vals), data[var].attrs)

    print('adding ensemble temperatures...')
    ens_temp_files = [os.path.join(utils.ERA_ens_temp_source, i) for i in sorted(os.listdir(utils.ERA_ens_temp_source))]
    
    with xr.open_mfdataset(sorted(ens_temp_files), combine='by_coords') as data:
        data = data.rename({'level': 'ens_level'})
        #ds.coords['number'] = data.coords['number']
        #ds.coords['ens_level'] = data.coords['ens_level']
        
        for var in data.data_vars.keys():
            var_shape = data[var].isel(time=0, latitude=0, longitude=0).shape
            vals = []
            for (lat, lon, time) in zip(lats, lons, times):
                if lat > np.max(data.coords['latitude']) or lat < np.min(data.coords['latitude']) or \
                    lon > np.max(data.coords['longitude']) or lon < np.min(data.coords['longitude']):
                    print('out of range of data')
                    print(lat, lon, time)
                    print(np.max(data.coords['latitude'].values), np.min(data.coords['latitude'].values))
                    print(np.max(data.coords['longitude'].values), np.min(data.coords['longitude'].values))
                    vals.append(np.full(var_shape, float('nan'), dtype='float'))
                    continue
                x = data[var].sel(longitude=slice(lon - box_degrees/2, lon + box_degrees/2),
                                  latitude=slice(lat + box_degrees/2, lat - box_degrees/2))
                z = x.sel(method='nearest', time=time, tolerance=np.timedelta64(1, 'h'))
                #this applies a 2D gaussian the width of z, i.e. sigma=box_degrees
                gauss_shape = tuple([v for v,i in zip(z.shape,z.dims) if i in ['latitude', 'longitude'] ])
                gauss = utils.gauss2D(shape=gauss_shape, sigma=gauss_shape[-1])
                filtered = z * gauss
                vals.append(filtered.sum(dim=('latitude', 'longitude')).values)
            ds['ERA_ens_'+var] = (tuple(x for x in data[var].dims if x not in ['latitude', 'longitude']), np.array(vals), data[var].attrs)
    return ds


def add_ERA_to_trajectory(ds, box_degrees=2):
    """Retrieve ERA5 data in a box around a trajectory
    Assumes ERA5 data is 0.3x0.3 degrees
    Returns an xarray Dataset
    """
    lats, lons, times = ds.lat.values, ds.lon.values, ds.time.values
    space_index = int(np.round(box_degrees/0.3/2)) # go up/down/left/right this many pixels
    unique_days = set([utils.as_datetime(i).date() for i in times])
    files = [os.path.join(utils.ERA_source, "ERA5.pres.NEP.{:%Y-%m-%d}.nc".format(i))
             for i in unique_days]
    flux_files = [os.path.join(utils.ERA_source, "ERA5.flux.NEP.{:%Y-%m-%d}.nc".format(i))
         for i in unique_days]
    with xr.open_mfdataset(sorted(files), combine='by_coords') as data:
        #return_ds = xr.Dataset(coords={'time': ds.coords['time'], 'level': data.coords['level']})
        ds.coords['level'] = data.coords['level']
        
        #adding in q:
        T = data['t'].values 
        RH = data['r'].values
        p = np.broadcast_to(data.coords['level'].values[None, :, None, None], T.shape)*100
        q = utils.qv_from_p_T_RH(p, T, RH)
        data['q'] = (('time', 'level', 'latitude', 'longitude'), q)
        data['q'] = data['q'].assign_attrs({'units': "kg kg**-1", 
                                'long_name': "specific_humidity",
                                'dependencies': 'ERA_t, ERA_p, ERA_r'})
        MR = q/(1-q)
        data['MR'] = (('time', 'level', 'latitude', 'longitude'), MR)
        data['MR'] = data['MR'].assign_attrs({'units': "kg kg**-1", 
                                'long_name': "mixing_ratio",
                                'dependencies': 'ERA_t, ERA_p, ERA_r'})
        
        # adding gradients in for z, t, and q. Assuming constant grid spacing.
        for var in ['t', 'q', 'z', 'u', 'v', 'MR']:
            [_,_,dvardj, dvardi] = np.gradient(data[var].values)
            dlatdy = 360/4.000786e7  # degrees lat per meter y
            def get_dlondx(lat) : return(360/(np.cos(np.deg2rad(lat))*4.0075017e7))

            lat_spaces = np.diff(data.coords['latitude'].values)
            lon_spaces = np.diff(data.coords['longitude'].values)
            assert(np.allclose(lat_spaces, -0.3, atol=0.01) and np.allclose(lon_spaces, 0.3, atol=0.05))
            dlondi = np.mean(lon_spaces)
            dlatdj = np.mean(lat_spaces)
            dlondx = get_dlondx(data.coords['latitude'].values)
            dvardx = dvardi/dlondi*dlondx[None,None,:,None]
            dvardy = dvardj/dlatdj*dlatdy
            data['d{}dx'.format(var)] = (('time', 'level', 'latitude', 'longitude'), dvardx)
            data['d{}dy'.format(var)] = (('time', 'level', 'latitude', 'longitude'), dvardy)

        grad_attrs = {'q': {'units': "kg kg**-1 m**-1",
                            'long_name': "{}_gradient_of_specific_humidity",
                            'dependencies': "ERA_t, ERA_p, ERA_r"},
                      't':  {'units': "K m**-1",
                            'long_name': "{}_gradient_of_temperature",
                            'dependencies': "ERA_t"},
                      'z':  {'units': "m**2 s**-2 m**-1",
                            'long_name': "{}_gradient_of_geopotential",
                            'dependencies': "ERA_z"},
                      'u': {'units': "m s**-1 m**-1",
                            'long_name': "{}_gradient_of_zonal_wind",
                            'dependencies': "ERA_u"},
                      'v': {'units': "m s**-1 m**-1",
                            'long_name': "{}_gradient_of_meridional_wind",
                            'dependencies': "ERA_v"},
                      'MR': {'units': "kg kg**-1 m**-1",
                            'long_name': "{}_gradient_of_mixing_ratio",
                            'dependencies': "ERA_t, ERA_p, ERA_r"}}

        for key, val in grad_attrs.items():
            for (n, drn) in [('x', 'eastward'), ('y', 'northward')]:
                attrs = val.copy()
                var = 'd{}d{}'.format(key, n)
                attrs['long_name'] = attrs['long_name'].format(drn)
                data[var] = data[var].assign_attrs(attrs)
            
        for var in data.data_vars.keys():
            vals = []
            for (lat, lon, time) in zip(lats, lons%360, times):
                if lat > np.max(data.coords['latitude']) or lat < np.min(data.coords['latitude']) or \
                    lon > np.max(data.coords['longitude']) or lon < np.min(data.coords['longitude']):
                    print(f'out of range of data" {lat}, {lon}, {time}')
                    vals.append(np.full_like(data.coords['level'], float('nan'), dtype='float'))
                    continue
                x = data[var].sel(longitude=slice(lon - box_degrees/2, lon + box_degrees/2),
                                  latitude=slice(lat + box_degrees/2, lat - box_degrees/2))

                z = x.sel(method='nearest', tolerance=np.timedelta64(1, 'h'), time=time)
                #z = y.sel(method='nearest', tolerance=50, level=pres)
                #this applies a 2D gaussian the width of z, i.e. sigma=box_degrees
                gauss = utils.gauss2D(shape=z.shape[1:], sigma=z.shape[0])
                filtered = z.values * gauss
                vals.append(np.sum(filtered, axis=(1,2)))
            ds['ERA_'+var] = (('time', 'level'), np.array(vals))
            ds['ERA_'+var] = ds['ERA_'+var].assign_attrs(data[var].attrs)
             
        t_1000 = ds.ERA_t.sel(level=1000).values
        theta_700 = met_utils.theta_from_p_T(p=700, T=ds.ERA_t.sel(level=700).values)
        LTS = theta_700-t_1000
        ds['ERA_LTS'] = (('time'), np.array(LTS))
        ds['ERA_LTS'] = ds['ERA_LTS'].assign_attrs(
                {"long_name": "Lower tropospheric stability",
                 "units": "K",
                 "_FillValue": "NaN"})
        t_dew = t_1000-(100-ds.ERA_r.sel(level=1000).values)/5
        lcl = met_utils.get_LCL(t=t_1000, t_dew=t_dew, z=ds.ERA_z.sel(level=1000).values/9.81)
        z_700 = ds.ERA_z.sel(level=700).values/9.81
        gamma_850 = met_utils.get_moist_adiabatic_lapse_rate(ds.ERA_t.sel(level=850).values, 850)
        eis = LTS - gamma_850*(z_700-lcl)
        ds['ERA_EIS'] = (('time'), np.array(eis))
        ds['ERA_EIS'] = ds['ERA_EIS'].assign_attrs(
                {"long_name": "Estimated inversion strength",
                 "units": "K",
                 "_FillValue": "NaN"})
        
        with xr.open_mfdataset(sorted(flux_files), combine='by_coords') as flux_data:
            for var in flux_data.data_vars.keys():
                vals = []
                for (lat, lon, time) in zip(lats, lons%360, times):
                    if lat > np.max(flux_data.coords['latitude']) or lat < np.min(flux_data.coords['latitude']) or \
                        lon > np.max(flux_data.coords['longitude']) or lon < np.min(flux_data.coords['longitude']):
                        print(f'out of range of data" {lat}, {lon}, {time}')
                        vals.append(float('nan'))
                        continue
                    x = flux_data[var].sel(longitude=slice(lon - box_degrees/2, lon + box_degrees/2),
                                          latitude=slice(lat + box_degrees/2, lat - box_degrees/2))
                    z = x.sel(method='nearest', time=time, tolerance=np.timedelta64(1, 'h'))
                    gauss = utils.gauss2D(shape=z.shape, sigma=z.shape[0])
                    filtered = z.values * gauss
                    vals.append(np.sum(filtered))
                ds['ERA_'+var] = (('time'), np.array(vals))
                ds['ERA_'+var] = ds['ERA_'+var].assign_attrs(flux_data[var].attrs)

    ds.attrs['ERA_params'] = f'ERA5 data acquired from ECWMF Copernicus at cds.climate.copernicus.eu/. statistics computed over a {box_degrees}-deg average centered on trajectory. EIS and LTS computed according to Wood and Bretherton (2006) and Klein and Hartmann (1993) respectively.'
    ds.attrs['ERA_reference'] = 'Copernicus Climate Change Service (C3S) (2017): ERA5: Fifth generation of ECMWF atmospheric reanalyses of the global climate . Copernicus Climate Change Service Climate Data Store (CDS), date of access. https://cds.climate.copernicus.eu/cdsapp#!/home'
         
    return ds
    

def add_MERRA_to_trajectory(ds, box_degrees=2, location='nep'):
    """Add MERRA-inferred aerosol number concentrations to trajectory.
    """ 
    lats, lons, times = ds.lat.values, ds.lon.values, utils.as_datetime(ds.time.values)
    unique_days = set([utils.as_datetime(i).date() for i in times])
    if location=='nep':
        files = [os.path.join("/home/disk/eos4/jkcm/Data/MERRA/3h/", "more_vertical", "MERRA2_400.inst3_3d_aer_Nv.{:%Y%m%d}.nc4.nc4".format(i))
                for i in unique_days]
    
    elif location=='sea':
        files = [os.path.join("/home/disk/eos4/jkcm/Data/MERRA/sea/new/", "MERRA2_400.inst3_3d_aer_Nv.{:%Y%m%d}.SUB.nc".format(i))
                for i in unique_days]

    with xr.open_mfdataset(sorted(files), combine='by_coords') as merra_data:
        
        if np.abs(np.mean(ds.lon.values))>90: # i.e. our trajectory is closer to 180 than it is to 0 lon. force to 0-360
            merra_data.coords['lon'] = merra_data['lon']%360
            lons = lons%360
        else:
            merra_data.coords['lon'] = (merra_data['lon']+180)%360-180
            lons = (lons+180)%360-180

        merra_data = merra_data.sel(lat=slice(np.min(lats)-2, np.max(lats)+2), lon=slice(np.min(lons)-2, np.max(lons)+2))

        
        # calculating MERRA pressure levels and heights
        dz = merra_data.DELP/(9.81*merra_data.AIRDENS)
        assert(merra_data.DELP.dims[1]=='lev')
        assert(dz.dims[1]=='lev')
        assert(dz.lev[-1]==72)     
        z = dz[:,::-1,:,:].cumsum(axis=1)[:,::-1,:,:]
        z.load()
        z[:, :-1, :, :] = (z.values[:, 1:, :, :]+z.values[:, :-1, :, :])/2
        z[:, -1,:,:] = z[:, -1,:,:]/2
        p = -merra_data.DELP[:,::-1,:,:].cumsum(axis=1)[:,::-1,:,:]
        p = p + merra_data.PS
        p.load()
        p[:, :-1, :, :] = (p.values[:, 1:, :, :]+p.values[:, :-1, :, :])/2
        p[:, -1,:,:] = (p[:, -1,:,:]+merra_data.PS)/2

        merra_data['H'] = z
        merra_data.H.attrs = {'long_name': 'mid_layer_heights', 'units': 'm'}
        merra_data['PL'] = p
        merra_data.PL.attrs = {'long_name': 'mid_level_pressure', 'units': 'Pa'}

        
        vals_to_add = ['Na_tot', 'MERRA_Na_tot_corr', 'H', 'PL', 'RH', 'AIRDENS']
        na_tot = np.zeros_like(merra_data.SS001.values)
        
        new_vals = []
        for varname,params in les_utils.merra_species_dict_colarco.items():
            vals_to_add.append(varname)
            var = merra_data[varname]

            num=les_utils.mass_to_number(mass=var, air_density=merra_data.AIRDENS.values, shape_params=params)

            na_tot = na_tot+num
            merra_data[varname+'_Na'] = (('time', 'lev', 'lat', 'lon'), num)
            new_vals.append(varname+'_Na')
        merra_data['Na_tot'] = (('time', 'lev', 'lat', 'lon'), na_tot, {'long_name': 'total aerosol number concentration, >100 um', 'units': 'cm**-3'})

        merra_data['MERRA_Na_tot_corr'] = (('time', 'lev', 'lat', 'lon'), np.exp(1.24*np.log(na_tot) + 0.18), {'long_name': 'total aerosol number concentration, >100 um, corrected to aircraft', 'units': 'cm**-3'})  
        merra_data['MERRA_Na_tot_corr_BL_logfit'] = (('time', 'lev', 'lat', 'lon'), np.exp(0.63*np.log(na_tot) + 2.42), {'long_name': 'total aerosol number concentration, >100 um, corrected to aircraft (boundary layer obs only)', 'units': 'cm**-3'})  
                   
                   
        ds = ds.assign_coords(lev = les_utils.MERRA_lev(merra_data.lev))
        
        merra_data = merra_data.assign_coords(lev = les_utils.MERRA_lev(merra_data.lev))                
        
        for var in vals_to_add+new_vals:
            var_shape = merra_data[var].isel(time=0, lat=0, lon=0).shape
            vals = []
            for (lat, lon, time) in zip(lats, lons, times):
                if lat > np.max(merra_data.coords['lat']) or lat < np.min(merra_data.coords['lat']) or \
                    lon > np.max(merra_data.coords['lon']) or lon < np.min(merra_data.coords['lon']):
                    print(f'out of range of data" {lat}, {lon}, {time}')
                    print(merra_data.coords['lat'])
                    print(merra_data.coords['lon'])
                    raise ValueError()
                    continue
                try:
                    x = merra_data[var].sel(lon=slice(lon - box_degrees/2, lon + box_degrees/2),
                                            lat=slice(lat - box_degrees/2, lat + box_degrees/2))
                except KeyError as e:
                    print(var)
                    print(lon, lat)
                    print(merra_data.lon)
                    print(merra_data.lat)

                    raise e
                z = x.sel(method='nearest', time=time, tolerance=np.timedelta64(2, 'h'))
                #this applies a 2D gaussian the width of z, i.e. sigma=box_degrees
                gauss_shape = tuple([v for v,i in zip(z.shape,z.dims) if i in ['lat', 'lon'] ])
                gauss = utils.gauss2D(shape=gauss_shape, sigma=gauss_shape[-1])
                filtered = z * gauss
                vals.append(filtered.sum(dim=('lat', 'lon')).values)
            if var in vals_to_add:
                attrs = merra_data[var].attrs
            elif var in new_vals:
                attrs = {'long_name': merra_data[var[:-3]].long_name + ', inferred aerosol number concentration', 'units':'cm**-3'}
            ds['MERRA_'+var] = (tuple(x for x in merra_data[var].dims if x not in ['lat', 'lon']), np.array(vals), attrs)
    
    
    
    
    ds['MERRA_Na_tot_mass'] = ds.MERRA_OCPHILIC + ds.MERRA_OCPHOBIC + ds.MERRA_BCPHILIC + \
                              ds.MERRA_BCPHOBIC + ds.MERRA_SO4 + ds.MERRA_DU001 + ds.MERRA_DU002 +\
                              ds.MERRA_DU003 +ds.MERRA_DU004 + ds.MERRA_DU005 + ds.MERRA_SS001 + \
                              ds.MERRA_SS002 + ds.MERRA_SS003 + ds.MERRA_SS004 + ds.MERRA_SS005
   
    # #akn=aitken = everything below 80nm
    # #acc = accumulution = everything between 80 and 1000
    # #crs=coarse = everything above 1000
    mass_acc_dict = {}
    mass_akn_dict = {}
    mass_crs_dict = {}
    num_acc_dict = {}
    num_akn_dict = {}
    num_crs_dict = {}
    
    
    
    for x in ['MERRA_OCPHILIC', 'MERRA_OCPHOBIC', 'MERRA_BCPHILIC', 'MERRA_BCPHOBIC', 'MERRA_SO4']:
        params = les_utils.merra_species_dict_colarco[x.split('_')[1]]
        data = ds[x]

        rho = ds.MERRA_AIRDENS.values 
        n0 = les_utils.get_n0(mass=data.values, density=params['density'], r_max=50, r_min=0.001, 
                              std_dev=params['geometric_std_dev'], mode_radius=params['mode_radius'])
        mass_acc_dict[x] = les_utils.get_m_subset(density=params['density'], n0=n0, r_min=0.08, r_max=1, 
                                                  std_dev=params['geometric_std_dev'], mode_radius=params['mode_radius'])
        mass_akn_dict[x] = les_utils.get_m_subset(density=params['density'], n0=n0, r_min=0.01, r_max=0.08, 
                                                  std_dev=params['geometric_std_dev'], mode_radius=params['mode_radius'])
        mass_crs_dict[x] = les_utils.get_m_subset(density=params['density'], n0=n0, r_min=1, r_max=params['upper'], 
                                                  std_dev=params['geometric_std_dev'], mode_radius=params['mode_radius'])
        num_acc_dict[x] = les_utils.get_n_subset(n0, r_min=0.08, r_max=1, 
                                                 std_dev=params['geometric_std_dev'], mode_radius=params['mode_radius'])
        num_akn_dict[x] = les_utils.get_n_subset(n0, r_min=0.01, r_max=0.08, 
                                                 std_dev=params['geometric_std_dev'], mode_radius=params['mode_radius'])
        num_crs_dict[x] = les_utils.get_n_subset(n0, r_min=1, r_max=params['upper'], 
                                                 std_dev=params['geometric_std_dev'], mode_radius=params['mode_radius'])
    
        ds[x+'_n0'] = (('time', 'lev'), n0*rho)


    mass_acc_attrs = {'long_name': 'accumulation mode aerosol mass',
                        'units': 'kg kg**-1'}
    mass_akn_attrs = {'long_name': 'aikten mode aerosol mass',
                        'units': 'kg kg**-1'}
    mass_crs_attrs = {'long_name': 'coarse mode aerosol mass',
                        'units': 'kg kg**-1'}

    ds['MERRA_acc_mass'] = (('time', 'lev'), np.sum(list(mass_acc_dict.values()), axis=0) + \
                                ds.MERRA_DU001.values + ds.MERRA_SS002.values + ds.MERRA_SS003.values,
                                mass_acc_attrs)
    ds['MERRA_akn_mass'] = (('time', 'lev'), np.sum(list(mass_akn_dict.values()), axis=0) + \
                                ds.MERRA_SS001.values,
                                mass_akn_attrs)
    ds['MERRA_crs_mass'] = (('time', 'lev'), np.sum(list(mass_crs_dict.values()), axis=0) + \
                                ds.MERRA_DU002.values + ds.MERRA_DU003.values + ds.MERRA_DU004.values + ds.MERRA_DU005.values + \
                                ds.MERRA_SS004.values + ds.MERRA_SS005.values,
                                mass_crs_attrs)
    
    num_acc_attrs = {'long_name': 'accumulation mode aerosol number',
                        'units': 'kg kg**-1'}
    num_akn_attrs = {'long_name': 'aikten mode aerosol number',
                        'units': 'kg kg**-1'}
    num_crs_attrs = {'long_name': 'coarse mode aerosol number',
                        'units': 'kg kg**-1'}
    
    ds['MERRA_acc_num'] = (('time', 'lev'), np.sum(list(num_acc_dict.values()), axis=0) + \
                                ds.MERRA_DU001_Na.values + ds.MERRA_SS002_Na.values + ds.MERRA_SS003_Na.values,
                                mass_acc_attrs)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    ds.lev.attrs['long_name'] = 'model level pressure'
    ds.lev.attrs['units'] = 'millibars'
    
    ds.attrs['MERRA_params'] = f'MERRA-2 data primarily downloaded from NASA GMAO, and statistics computed over a {box_degrees}-deg average centered on trajectory. For aerosol estimates (Na), equivalent aerosol number is computed based on aerosol mass consistent with the MERRA2-assumed aerosol optical properties. Contact jkcm@uw.edu for details.'
    ds.attrs['MERRA_reference'] = 'MERRA-2 data available at https://gmao.gsfc.nasa.gov/reanalysis/MERRA-2.'
    
    return ds


def new_add_MERRA_to_trajectory(ds, box_degrees=2):
    
    ds['MERRA_Na_tot_mass'] = ds.MERRA_OCPHILIC + ds.MERRA_OCPHOBIC + ds.MERRA_BCPHILIC + ds.MERRA_BCPHOBIC + ds.MERRA_SO4 + ds.MERRA_DU001 + ds.MERRA_DU002 + ds.MERRA_DU003 +ds.MERRA_DU004 + \
                       ds.MERRA_DU005 + ds.MERRA_SS001 + ds.MERRA_SS002 + ds.MERRA_SS003 + ds.MERRA_SS004 + ds.MERRA_SS005
    
    #akn=aitken = everything below 80nm
    #acc = accumulution = everything between 80 and 1000
    #crs=coarse = everything above 1000
    
    mass_acc_dict = {}
    mass_aik_dict = {}
    mass_crs_dict = {}
    num_acc_dict = {}
    num_aik_dict = {}
    num_crs_dict = {}
    
    # for x in ['MERRA_OCPHILIC', 'MERRA_OCPHOBIC', 'MERRA_BCPHILIC', 'MERRA_BCPHOBIC', 'MERRA_SO4']:
        
    
    # ds['MERRA_Na_acc_mass'] = 
    # ds['MERRA_Na_akn_mass'] = 
    # ds['MERRA_Na_crs_mass'] = 
    
    # ds['MERRA_akn_num'] = 
    # ds['MERRA_acc_num'] = 
    # ds['MERRA_crs_num'] = 
    
    
    
    
    return ds

    #aitken = low-100nm
    #add aikten NUMBER
    #add aikten mass
    

def add_speeds_to_trajectories(ds):
    """Add speed variables to trajectory. used centered difference of distances traveled
    """
    lats, lons, times = ds.lat.values, ds.lon.values, ds.time.values
    
    heading_starts, heading_ends, seg_speeds = [], [], []
    
    for i in range(len(lats)-1):
        geod = Geodesic.WGS84.Inverse(lats[i], lons[i], lats[i+1], lons[i+1])
        dtime = (times[i+1]-times[i])/np.timedelta64(1, 's')
        heading_starts.append(geod['azi1'])
        heading_ends.append(geod['azi2'])
        seg_speeds.append(geod['s12']/dtime)

    #speeds are centered difference, except at start and end, where they are speeds of 
    #first and last trajectory segments
    #headings are average of end azimuth of previous segment/start azimuth of next geodesic segment,
    #except at start and end, where are just the start/end azimuths of the first/last geodesic
    speeds = np.mean(np.vstack([seg_speeds+[seg_speeds[-1]],[seg_speeds[0]]+seg_speeds]), axis=0)
     #headings = np.mean(np.vstack([[heading_starts[0]]+heading_ends, heading_starts+[heading_ends[-1]]]), axis=0) THIS HAD A BUG
    def radial_mean(h1, h2):
        diff = ((h2-h1)+180)%360-180
        return h1 + diff/2
    headings = radial_mean(np.array([heading_starts[0]]+heading_ends), np.array(heading_starts+[heading_ends[-1]]))
    
    u = speeds*np.cos(np.deg2rad(90-headings))
    v = speeds*np.sin(np.deg2rad(90-headings))
    
    ds['traj_u'] = (('time'), u, {'long_name': 'U component of trajectory velocity', 'units': "m s**-1"})
    ds['traj_v'] = (('time'), v, {'long_name': 'V component of trajectory velocity', 'units': "m s**-1"})
    ds['traj_hdg'] = (('time'), headings, {'long_name': 'Trajectory heading', 'units': 'deg'})
    ds['traj_spd'] = (('time'), speeds, {'long_name': 'Trajectory speed', 'units': "m s**-1"})
    return ds


def add_advection_to_trajectory(ds):
    """Add advection to trajectory after adding ERA data
    """
    names = dict(u='ERA_u', v='ERA_v', u_t='traj_u', v_t='traj_v',
                   dtdx='ERA_dtdx', dtdy='ERA_dtdy', dqdx='ERA_dqdx', dqdy='ERA_dqdy', dMRdx='ERA_dMRdx', dMRdy='ERA_dMRdy')
    assert np.all([i in ds.data_vars.keys() for i in names.values()])
    rel_adv_of_T = -((ds[names['u']].values-ds[names['u_t']].values[:, None])*ds[names['dtdx']].values + \
                   (ds[names['v']].values-ds[names['v_t']].values[:, None])*ds[names['dtdy']].values)
    rel_adv_of_q = -((ds[names['u']].values-ds[names['u_t']].values[:, None])*ds[names['dqdx']].values + \
                   (ds[names['v']].values-ds[names['v_t']].values[:, None])*ds[names['dqdy']].values)
    rel_adv_of_MR = -((ds[names['u']].values-ds[names['u_t']].values[:, None])*ds[names['dMRdx']].values + \
                   (ds[names['v']].values-ds[names['v_t']].values[:, None])*ds[names['dMRdy']].values)
    T_adv_attr = {'units': "K s**-1", 
                  'long_name': "trajectory_relative_advection_of_temperature",
                  'dependencies': 'ERA_t, traj_u, traj_v, ERA_u, ERA_v'}
    q_adv_attr = {'units': "kg kg**-1 s**-1", 
                  'long_name': "trajectory_relative_advection_of_specific_humidity",
                  'dependencies': 'ERA_q, traj_u, traj_v, ERA_u, ERA_v'}
    MR_adv_attr = {'units': "kg kg**-1 s**-1", 
                  'long_name': "trajectory_relative_advection_of_mixing ratio",
                  'dependencies': 'ERA_q, traj_u, traj_v, ERA_u, ERA_v'}
        
    ds['ERA_T_adv'] = (('time', 'level'), rel_adv_of_T)
    ds['ERA_T_adv'] = ds['ERA_T_adv'].assign_attrs(**T_adv_attr)
    
    ds['ERA_q_adv'] = (('time', 'level'), rel_adv_of_q)
    ds['ERA_q_adv'] = ds['ERA_q_adv'].assign_attrs(**q_adv_attr)
    
    ds['ERA_MR_adv'] = (('time', 'level'), rel_adv_of_MR)
    ds['ERA_MR_adv'] = ds['ERA_MR_adv'].assign_attrs(**MR_adv_attr)
    return ds


def add_upwind_profile_to_trajectory(ds, dist=200, box_avg=2):
    """Add 'upwind' profile (not a true profile since the location varies with height)
    for alternative nudging method.
    Add only T_upwind, q_upwind, and MR_upwind vars
    """
    T_upwind = np.full_like(ds.ERA_t, np.nan)
    q_upwind = np.full_like(ds.ERA_q, np.nan)
    MR_upwind = np.full_like(ds.ERA_MR, np.nan)
    
    for i, t in enumerate(ds.time):
        for j, l in enumerate(ds.level):
            u = ds.ERA_u.sel(time=t, level=l)
            v = ds.ERA_v.sel(time=t, level=l)
    
    
def add_ERA_sfc_data(ds, box_degrees=2):
    
    lats, lons, times = ds.lat.values, ds.lon.values, ds.time.values
    space_index = int(np.round(box_degrees/0.3/2)) # go up/down/left/right this many pixels
    unique_days = set([utils.as_datetime(i).date() for i in times])
    sfc_files = [os.path.join(utils.ERA_source, "ERA5.sfc.NEP.{:%Y-%m-%d}.nc".format(i))
             for i in unique_days]
    with xr.open_mfdataset(sorted(sfc_files), combine='by_coords') as data:
         for var in data.data_vars.keys():
            vals = []
            for (lat, lon, time) in zip(lats, lons%360, times):
                if lat > np.max(data.coords['latitude']) or lat < np.min(data.coords['latitude']) or \
                    lon > np.max(data.coords['longitude']) or lon < np.min(data.coords['longitude']):
                    print('out of range of data')
                    print(lat, lon, time)
                    vals.append(float('nan'))
                    continue
                x = data[var].sel(longitude=slice(lon - box_degrees/2, lon + box_degrees/2),
                                  latitude=slice(lat + box_degrees/2, lat - box_degrees/2))
                z = x.sel(method='nearest', tolerance=np.timedelta64(59, 'm'), time=time)
                gauss = utils.gauss2D(shape=z.shape, sigma=z.shape[0])
                filtered = z.values * gauss
                vals.append(np.sum(filtered))
            ds['ERA_'+var] = (('time'), np.array(vals))
            ds['ERA_'+var] = ds['ERA_'+var].assign_attrs(data[var].attrs)
            
    #lhf = ds['ERA_ie'].values*2264705
    #ds['ERA_ilhf'] = (('time'), lhf)
    #ds['ERA_ilhf'] = ds['ERA_ilhf'].assign_attrs({"long_name": "Instantaneous surface latent heat flux",
    #                                              "units": "W m**-2",
    #                                              "_FillValue": "NaN"})
    #ds['ERA_'+var] = ds['ERA_'+var]

    return ds


def add_MODISPBL_to_trajectory(ds, box_degrees=3):
    lats, lons, times = ds.lat.values, ds.lon.values, ds.time.values
    MODIS_day_idx = np.argwhere([i.hour == 23 for i in utils.as_datetime(times)]).squeeze()
    MODIS_night_idx = np.argwhere([i.hour == 11 for i in utils.as_datetime(times)]).squeeze()
     #dayfile = '/home/disk/eos4/jkcm/Data/CSET/Ryan/Daily_1x1_JHISTO_CTH_c6_day_v2_calboxes_top10_Interp_hif_zb_2011-2016.nc'
    dayfile = '/home/disk/eos4/jkcm/Data/CSET/Ryan/Daily_1x1_JHISTO_CTH_c6_day_v2_calboxes_top10_Interp_hif_zb_2011-2016_corrected.nc'
    nightfile = '/home/disk/eos4/jkcm/Data/CSET/Ryan/Daily_1x1_JHISTO_CTH_c6_night_v2_calboxes_top10_Interp_hif_zb_2011-2016.nc'
    vals = []   
    stds = []
    nanfrac = []
    medians = []
    counts = []
    mins = []
    maxs = []

    for i in range(len(times)):
        if i in MODIS_day_idx:
            f = dayfile
        elif i in MODIS_night_idx:
            f = nightfile
        else:
            vals.append(np.nan)
            stds.append(np.nan)
            nanfrac.append(np.nan)
            medians.append(np.nan)
            counts.append(np.nan)
            mins.append(np.nan)
            maxs.append(np.nan)
            continue
        with xr.open_dataset(f) as data:
            lat, lon, time = lats[i], lons[i], utils.as_datetime(times[i])
            t_idx = np.argwhere(np.logical_and(data['days'].values == time.timetuple().tm_yday, 
                                   data['years'].values == time.year))[0][0]
            x = data['cth'].sel(longitude=slice(lon - box_degrees/2, lon + box_degrees/2),
                  latitude=slice(lat + box_degrees/2, lat - box_degrees/2))
            z = x.isel(time=t_idx).values
            vals.append(np.nanmean(z))
            medians.append(np.nanmedian(z))
            counts.append(np.sum(~np.isnan(z)))
            stds.append(np.nanstd(z))
            nanfrac.append(np.sum(np.isnan(z))/z.size)
            mins.append(np.nanmin(z))
            maxs.append(np.nanmax(z))
    ds['MODIS_CTH'] = (('time'), np.array(vals), {'long_name': 'MODIS cloud top height, box mean', 'units': 'km'})
    ds['MODIS_CTH_median'] = (('time'), np.array(medians), {'long_name': 'MODIS cloud top height, box median', 'units': 'km'})
    ds['MODIS_CTH_std'] = (('time'), np.array(stds), {'long_name': 'MODIS cloud top height, box standard deviation', 'units': 'km'})
    ds['MODIS_CTH_min'] = (('time'), np.array(mins), {'long_name': 'MODIS cloud top height, box min', 'units': 'km'})
    ds['MODIS_CTH_max'] = (('time'), np.array(maxs), {'long_name': 'MODIS cloud top height, box max', 'units': 'km'})
    ds['MODIS_CTH_nanfrac'] = (('time'), np.array(nanfrac), {'long_name': 'MODIS missing pixel fraction', 'units': '0-1'})
    ds['MODIS_CTH_n_samples'] = (('time'), np.array(counts), {'long_name': 'MODIS CTH number of samples'})
    
    
    
    ds.attrs['MODIS_params'] = f'MODIS Aqua cloud-top height from Eastman et al. (2017) computed over a {box_degrees}-deg average centered on trajectory'
    ds.attrs['MODIS_reference'] = f"Eastman, R., Wood, R., & O, K. T. (2017). The Subtropical Stratocumulus-Topped Planetary Boundary Layer: A Climatology and the Lagrangian Evolution. Journal of the Atmospheric Sciences, 74(8), 2633–2656. https://doi.org/10.1175/JAS-D-16-0336.1"
    return ds


def add_SSMI_to_trajectory(ds, box_degrees=2):
    lats, lons, times = ds.lat.values, ds.lon.values, ds.time.values
    cloud_vals = np.full_like(lats, fill_value=np.nan)
    vapor_vals = np.full_like(lats, fill_value=np.nan)
    wspd_vals = np.full_like(lats, fill_value=np.nan)
    cloud_vals_std = np.full_like(lats, fill_value=np.nan)
    vapor_vals_std = np.full_like(lats, fill_value=np.nan)
    wspd_vals_std = np.full_like(lats, fill_value=np.nan)
    count_vals = np.full_like(lats, fill_value=np.nan)
    sats = ['f15', 'f16' ,'f17', 'f18']
    for sat in sats:
        ssmi_data = xr.open_mfdataset(glob.glob(f'/home/disk/eos9/jkcm/Data/ssmi/all/ssmi_unified_{sat}*.nc'), concat_dim='time', combine='by_coords')
        for i, (lat, lon, time) in enumerate(zip(lats, lons%360, times)):
            for orbit_segment in [0,1]:
                ds_sub = ssmi_data.sel(time=time-np.timedelta64(1-orbit_segment, 'D'), method='nearest', tolerance=np.timedelta64(24, 'h')).sel(orbit_segment=orbit_segment)
                #note to future users: because of the longitude of CSET, the 1-day offset is included. This is a hardcoded value to deal with the fact that the UTC
                # day of SSMI doesn't line up when you're near the antimeridian. Future use should deal with this better, by deciding whether or not to add a day
                # based on the longitude being considered. 
                ds_sub2 = ds_sub.sel(longitude=slice(lon - box_degrees/2, lon + box_degrees/2),
                                          latitude=slice(lat - box_degrees/2, lat + box_degrees/2))
                sel_date = ds_sub2.UTCtime
                nsample = 1-(np.sum(ds_sub2.nodata)/np.size(ds_sub2.nodata)).values
                if nsample < 0.5:
        #             print(f'skipping {time}')
                    continue
                else:

                    meantime = np.nanmean(ds_sub2.UTCtime.values)
                    sampletime = ds_sub2.time.values + np.timedelta64(int(meantime), 'h') + np.timedelta64(int(60*(meantime - int(meantime))), 'm')
                    miss = (time-sampletime)/np.timedelta64(1, 'h')
                    if np.abs(miss)<0.5:
                         #print(f'{sat}: found data at {time}, off by {miss} hours. sample fill is {nsample:.0%}')
                        # print(np.sum(~np.isnan(ds_sub2.cloud).values), np.sum(~np.isnan(ds_sub2.vapor).values), np.sum(~np.isnan(ds_sub2.wspd_mf)).values)
                        # print(np.size(ds_sub2.nodata.values)-np.sum(ds_sub2.nodata.values), np.sum(~ds_sub2.nodata).values)
                        cloud_vals[i] = np.nanmean(ds_sub2.cloud)
                        vapor_vals[i] = np.nanmean(ds_sub2.vapor)
                        wspd_vals[i] = np.nanmean(ds_sub2.wspd_mf)
                        count_vals[i] = np.sum(~np.isnan(ds_sub2.cloud))
                        cloud_vals_std[i] = np.nanstd(ds_sub2.cloud)
                        vapor_vals_std[i] = np.nanstd(ds_sub2.vapor)
                        wspd_vals_std[i] = np.nanstd(ds_sub2.wspd_mf)
        #             break
    ds['SSMI_LWP'] = (('time'), np.array(cloud_vals), ds_sub2.cloud.attrs)
    ds['SSMI_LWP_std'] = (('time'), np.array(cloud_vals_std), ds_sub2.cloud.attrs)
    ds['SSMI_WVP'] = (('time'), np.array(vapor_vals), ds_sub2.vapor.attrs)
    ds['SSMI_VWP_std'] = (('time'), np.array(vapor_vals_std), ds_sub2.vapor.attrs)
    ds['SSMI_WSPD'] = (('time'), np.array(wspd_vals), ds_sub2.wspd_mf.attrs)
    ds['SSMI_WSPD_std'] = (('time'), np.array(wspd_vals_std), ds_sub2.wspd_mf.attrs)
    ds['SSMI_n_samples'] = (('time'), np.array(count_vals), {'long_name': 'SSMI number of data samples'})
    ds['SSMI_WSPD_std'] = (('time'), np.array(wspd_vals_std), ds_sub2.wspd_mf.attrs)
    for i in ['SSMI_LWP_std', 'SSMI_VWP_std', 'SSMI_WSPD_std']:
        ds[i].attrs['long_name'] = ds[i].attrs['long_name']+' standard deviation over box'
        
    ds.attrs['SSMI_params'] = f'SSM/I data added from satellites {", ".join(sats).upper()}; statistics computed over a {box_degrees}-deg average centered on trajectory'
    ds.attrs['SSMI_reference'] = f"SSM/I and SSMIS data are produced by Remote Sensing Systems. Data are available at www.remss.com/missions/ssmi."

    return ds


def add_CERES_to_trajectory(ds, box_degrees=2):
    lats, lons, times = ds.lat.values, ds.lon.values, ds.time.values

    ceres_file = xr.open_mfdataset('/home/disk/eos9/jkcm/Data/ceres/proc/split/CERES_SYN1deg-1H_Terra-Aqua-MODIS_Ed4.*.nc', combine='by_coords')

    attr_dict ={'net_cre': {'long_name': 'Net Cloud Radiative Effect', 'units': 'W m-2'},
            'sw_cre': {'long_name': 'Shortwave Cloud Radiative Effect', 'units': 'W m-2'},
            'lw_cre': {'long_name': 'Longwave Cloud Radiative Effect', 'units': 'W m-2'}}
    
    means_dict, stds_dict = dict(), dict()
    for i, (lat, lon, time) in enumerate(zip(lats, lons%360, times)):
        ds_sub = ceres_file.sel(time=time, method='nearest', tolerance=np.timedelta64(24, 'h'))
        ds_sub2 = ds_sub.sel(lon=slice(lon - box_degrees/2, lon + box_degrees/2),
                             lat=slice(lat - box_degrees/2, lat + box_degrees/2))
        for var in  ['toa_sw_all_1h', 'toa_sw_clr_1h', 'toa_lw_all_1h', 'toa_lw_clr_1h', 'lw_cre', 'sw_cre', 'net_cre', 'toa_solar_all_1h', 'cldarea_low_1h' , 
                     'cldtau_low_1h', 'lwp_low_1h', 'solar_zen_angle_1h', 'cldwatrad_low_1h',
                     'adj_atmos_sw_down_all_surface_1h', 'adj_atmos_sw_up_all_surface_1h', 'adj_atmos_lw_down_all_surface_1h', 'adj_atmos_lw_up_all_surface_1h']:
            
            means_dict.setdefault(var, []).append(np.nanmean(ds_sub2[var]))
            stds_dict.setdefault(var, []).append(np.nanstd(ds_sub2[var]))
                                                  
    for var in means_dict.keys():
        attrs = attr_dict.setdefault(var, ds_sub2[var].attrs)
        ds['CERES_'+var] = (('time'), np.array(means_dict[var]), attrs)
        attrs.update(long_name=attrs['long_name']+', standard deviation over box')
        ds['CERES_'+var+'_std'] = (('time'), np.array(stds_dict[var]), attrs)
    
    #adding ND
    nd = 1.4067 * 10**4 * (ds.CERES_cldtau_low_1h.values ** 0.5) / (ds.CERES_cldwatrad_low_1h**2.5)
    ds['CERES_Nd'] = (('time'), nd, {'long_name': 'cloud droplet number concentration',
     'units': 'cm**-3'})
    ds.attrs['CERES_reference'] = ds.CERES_reference + '\n CERES Nd derived following Painemal and Zuidema (2011), eqn 7 (doi:10.1029/2011JD016155).'

    ds.attrs['CERES_params'] = f'CERES data is from SYN1deg hourly product; statistics computed over a {box_degrees}-deg average centered on trajectory'
    ds.attrs['CERES_reference'] = f'CERES data available from NASA LARC at ceres.larc.nasa.gov/data.  doi: 10.1175/JTECH-D-12-00136.1, doi: 10.1175/JTECH-D-15-0147.1'

    return ds
    

def add_amsr_to_trajectory(ds, box_degrees=2):     
    lats, lons, times = ds.lat.values, ds.lon.values, ds.time.values

    try:
        precip_data = xr.open_mfdataset(np.unique([f'/home/disk/eos9/jkcm/Data/rain/{i.year}/AMSR2_89GHz_pcp_est_{i.year}_{i.dayofyear:03}_day_gridded.nc' 
                                                   for i in pd.DatetimeIndex(ds.time.values)]) , combine='by_coords')
        
        means_dict, stds_dict = dict(), dict()
        for i, (lat, lon, time) in enumerate(zip(lats, lons%360, times)):
            ds_sub = precip_data.sel(date=time, method='nearest', tolerance=np.timedelta64(24, 'h'))  
            ds_sub2 = ds_sub.sel(longitude=slice(lon - box_degrees/2, lon + box_degrees/2),
                                 latitude=slice(lat - box_degrees/2, lat + box_degrees/2))
            # print(ds_sub2.time)       

            print((ds_sub2.time.values - time)/np.timedelta64(1, 'h'))

        

            for var in  ['rain_rate_mean', 'rain_rwr_mean', 'rain_prob_mean']:
                    
                    means_dict.setdefault(var, []).append(np.nanmean(ds_sub2[var]))
                    stds_dict.setdefault(var, []).append(np.nanstd(ds_sub2[var]))

            for var in means_dict.keys():
                attrs = attr_dict.setdefault(var, ds_sub2[var].attrs)
                ds['AMSR_TB_'+var] = (('time'), np.array(means_dict[var]), attrs)
                attrs.update(long_name=attrs['long_name']+', standard deviation over box')
                ds['AMSR_TB_'+var+'_std'] = (('time'), np.array(stds_dict[var]), attrs)

    except (FileNotFoundError, AttributeError) as e:
        raise e
    #     if isinstance(e, FileNotFoundError):
    #         print('could not find precip file for date, continuing:' + str(date))
    #         print(e)
    #     elif isinstance(e, AttributeError):
    #         print('attribute error, likely no data found' + str(date))
    #         print(e)
            
    #     df['amsr_tb_rate'] = np.nan
    #     df['amsr_tb_rwr'] = np.nan
    #     df['amsr_tb_prob'] = np.nan
        
    # except KeyError as e:
    #     print(precip_data.date)
    #     print(date)
    #     print(e)
    
    return ds


def make_trajectory(rfnum, trajnum, save=False, trajectory_type='500m_+72'):
    ds = xarray_from_trajectory(rfnum, trajnum, trajectory_type)
    ds = add_speeds_to_trajectories(ds)
    print("adding ERA...")
    ds = add_ERA_to_trajectory(ds)
    print('adding advection...')
    ds = add_advection_to_trajectory(ds)
    print('adding ERA sfc data...')
    ds = add_ERA_sfc_data(ds)
    #print('adding ERA ensemble data...')
    #ds = add_ERA_ens_to_trajectory(ds)
    print('adding GOES data...')
    ds = add_GOES_obs(ds)
    print("adding MODIS...")
    ds = add_MODISPBL_to_trajectory(ds)
    print("adding MERRA...")
    ds = add_MERRA_to_trajectory(ds)
    print("adding SSMI...")
    ds = add_SSMI_to_trajectory(ds)
    print("adding CERES...")
    ds = add_CERES_to_trajectory(ds)
    if save:
        save_trajectory_to_netcdf(ds, save)
    return ds


def update_trajectory(name):
    ds = xr.open_dataset(name)
    ds = new_add_MERRA_to_trajectory(ds)
    save_trajectory_to_netcdf(ds, name)
    

def add_AMSR_to_trajectory(ds, box_degrees=2):
    lats, lons, times = ds.lat.values, ds.lon.values%360, ds.time.values
    # daynight = {}
    amsr_prcp_mean = np.full(len(lats), fill_value=np.nan)
    amsr_prcp_std = np.full(len(lats), fill_value=np.nan)
    amsr_prcp_count = np.full(len(lats), fill_value=np.nan)
    amsr_prcp_dist = np.full((len(lats), 9), fill_value=np.nan)
    for dn in ['day', 'night']:
        # print(f'processing amsr {dn}time')
        precip_data = xr.open_mfdataset(np.unique([f"/home/disk/eos5/rmeast/rain_rates_89/{i.year}/AMSR2_89GHz_pcp_est_{i.year}_{i.dayofyear:03}_{dn}.nc" 
                                                    for i in pd.DatetimeIndex(ds.time.values)]), combine='nested', concat_dim='time')
        year = precip_data.time_vars.isel(yr_day_utc=0).values
        day = precip_data.time_vars.isel(yr_day_utc=1).values
        utc = precip_data.time_vars.isel(yr_day_utc=2).values
        total_secs = (utc*3600)
        secs = total_secs//1
        msecs = 1000*total_secs%1
        dtime = np.datetime64(f'{np.median(year):0.0f}-01-01')+np.timedelta64(1, 'D')*(day-1)+np.timedelta64(1, 's')*(secs)+np.timedelta64(1, 'ms')*(msecs)
        precip_data['time'] = (('time'), dtime)
        precip_data = precip_data.drop(labels=['time_vars'])
        precip_data['longitude'] = precip_data['longitude']%360


        # filtering out the misaligned times, couple scans at day start/end
        tdiff = np.diff(precip_data.time)/np.timedelta64(1, 'h')
        ups = np.argwhere(tdiff>20).flatten()
        downs = np.argwhere(tdiff<-20).flatten()
        if len(ups)<len(downs):
            if len(ups) == 0: # starting with a misalignment and only one error:
                ups = np.insert(ups, 0, 0) 
            elif ups[0]>downs[0] : # also starting with a misalignment; this is done in a kludgy way to avoid an index error
                ups = np.insert(ups, 0, 0) # chop off from 0 to first down
        good = np.ones(precip_data.time.shape).astype(bool)
        for (u,d) in zip(ups, downs):
            good[u:d+1] = np.zeros_like(good[u:d+1])
        precip_data = precip_data.isel(time=good)

        #actual statistics come here

        for i, (lat, lon, time) in enumerate(zip(lats, lons, times)):
            try:
                ds_sub = precip_data.sel(time=slice(time-np.timedelta64(1, 'h'), time+np.timedelta64(1, 'h')))
            except (ValueError, KeyError) as e:
                raise e
            good = np.logical_and(np.abs(ds_sub.latitude-lat)<1, np.abs(ds_sub.longitude-lon)<1)
            n_good = np.sum(good).values
            n_agood = 0
            if n_good > 100: # at least 100 actual pixels in the swath
                actual_good = ds_sub.rain_stats.isel(prob_rate_rwr_max=0).where(good)>0
                n_agood = np.sum(actual_good).values
                # print(f'{dn}: {i}, {time} inside swath')
            else:
                pass #outside of swath
            if n_agood > 100: # at least 100 pixels with a valid retrieval
                g_rate = ds_sub.rain_stats.isel(prob_rate_rwr_max=1).where(good).values.flatten()[good.values.flatten()]
                g_rate = g_rate[~np.isnan(g_rate)] # this should be redundant

                for var in [g_rate]:
                    var = var[~np.isnan(var)]
                    sorted_var = sorted(var)
                    cumsum = np.cumsum(sorted_var)
                    pctiles = np.linspace(0.1*np.sum(var), 0.9*np.sum(var), 9)
                    pct_vals = np.array([sorted_var[j] for j in [np.argmax(cumsum>=i) for i in pctiles]])

                    
                amsr_prcp_mean[i] = np.nanmean(g_rate)
                amsr_prcp_std[i] = np.nanstd(g_rate)
                amsr_prcp_count[i] = n_agood
                amsr_prcp_dist[i] = pct_vals
            else:
                pass # leave it as nans
            
    ds['AMSR_prcp_mean'] = (('time'), amsr_prcp_mean, {'long_name': 'AMSR Tb mean rain rate', 'units': 'mm hr^-1'})
    ds['AMSR_prcp_std'] = (('time'), amsr_prcp_std, {'long_name': 'AMSR Tb rain rate standard deviation', 'units': 'mm hr^-1'})
    ds['AMSR_prcp_n_samples'] = (('time'), amsr_prcp_count, {'long_name': 'AMSR Tb rain rate sample count', 'units': ''})
    ds['AMSR_prcp_dist'] = (('time', 'AMSR_prcp_pctile'), amsr_prcp_dist, {'long_name': 'AMSR Tb rain rate cumulative distribution', 'units': 'mm hr^-1'})
    ds['AMSR_prcp_pctile'] = (('AMSR_prcp_pctile'), np.linspace(10, 90, 9), {'long_name': 'rain below value in amsr_prcp_dist explains this much of all precipitation'})
    

    ds.attrs['AMSR_params'] = f'AMSR data statistics computed over a {box_degrees}-deg average centered on trajectory, only when > 100 samples are present'
    ds.attrs['AMSR_reference'] = f'AMSR precipitation is derived from 89 GHz brightness temperature, from Eastman, R., Lebsock, M., & Wood, R. (2019). Warm Rain Rates from AMSR-E 89-GHz Brightness Temperatures Trained Using CloudSat Rain-Rate Observations. Journal of Atmospheric and Oceanic Technology, 36(6), 1033–1051. https://doi.org/10.1175/JTECH-D-18-0185.1'

    return ds    
    
    
if __name__ == "__main__":

    
    force_override = True
    for case_num, case in utils.all_cases.items():
        print('working on case {}'.format(case_num))
        if case_num not in [6, 10]:
            continue
        flight = case['TLC_name'].split("_")[1][:4].lower()
        traj_list = case['TLC_name'].split('_')[2].split('-')
        for dirn in ['forward', 'backward']:
            nc_dirstring = '48h_backward' if dirn == 'backward' else '72h_forward'
            for traj in traj_list:
                #if traj not in ['2.0']:
                #    continue
                name = os.path.join(utils.trajectory_netcdf_dir, "{}_{}_{}.nc".format(flight, nc_dirstring, traj))
                print("working on {}...".format(os.path.basename(name)))
                if os.path.exists(name):
                    print("already exists!")
                    if not force_override:
                        continue
                    else:
                        print('overriding')
                        os.rename(name, os.path.join(utils.trajectory_netcdf_dir, 'old', "{}_{}_{}.nc".format(flight, nc_dirstring, traj)))
    #             ds = make_trajectory(rfnum=flight, trajnum=float(traj), save=name);
                trajectory_type = '500m_-48' if dirn == 'backward' else '500m_+72'
                ds = make_trajectory(rfnum=flight, trajnum=float(traj), save=name, trajectory_type=trajectory_type);


    #ds = add_ERA_sfc_data(ds)
    #ds = make_trajectory(rfnum='rf06', trajnum=2.3, save=False)
    #save_trajectory_to_netcdf(ds, r'/home/disk/eos4/jkcm/Data/CSET/model_forcings/rf06_traj_2.3_fullcolumn_withz.nc')

#     all_trajs = {'rf06': [1.6, 2.0, 2.3, 2.6, 3.0],
#                  'rf10': [5.5, 6.0]}


    # for flight, traj_list in all_trajs.items():
    #     for traj in traj_list:
    #         name = os.path.join(utils.trajectory_netcdf_dir, "{}_MODIS_traj_{:0.1f}.nc".format(flight, traj))
    #         print("working on {}...".format(os.path.basename(name)))
    #         ds = make_trajectory(rfnum=flight, trajnum=traj, save=name);


    # ds = make_trajectory(rfnum='rf06', trajnum=2.3, save=False)
    # save_trajectory_to_netcdf(ds, r'/home/disk/eos4/jkcm/Data/CSET/Lagrangian_project/trajectory_files/rf06_MODIS_traj_2.3.nc')


    # ds = make_trajectory(rfnum='rf10', trajnum=6.0, save=False)
    # save_trajectory_to_netcdf(ds, r'/home/disk/eos4/jkcm/Data/CSET/Lagrangian_project/trajectory_files/rf10_MODIS_traj_6.0.nc')

