import xarray as  xr
import numpy as np

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
        for var in  ['toa_alb_all_1h', 'toa_alb_clr_1h', 'toa_sw_all_1h', 'toa_sw_clr_1h', 'toa_lw_all_1h', 'toa_lw_clr_1h', 'lw_cre', 'sw_cre', 'net_cre', 'toa_solar_all_1h', 'cldarea_low_1h' , 
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
    ds.attrs['CERES_params'] = f'CERES data is from SYN1deg hourly product; statistics computed over a {box_degrees}-deg average centered on trajectory'
    ds.attrs['CERES_reference'] = f'CERES data available from NASA LARC at ceres.larc.nasa.gov/data.  doi: 10.1175/JTECH-D-12-00136.1, doi: 10.1175/JTECH-D-15-0147.1 \n CERES Nd derived following Painemal and Zuidema (2011), eqn 7 (doi:10.1029/2011JD016155).'
    return ds
    

