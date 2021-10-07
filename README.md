# uw-trajectory
University of Washington Trajectory Colocated Extraction

Maintainers:
Johannes Mohrmann (jkcm@uw.edu)

Code repository for the running of wind-driven trajectories, based on model or reanalysis gridded winds, and the colocation and extraction of geospatial datasets along trajectories. 

Todo: all of it!







Notes on geographic regions:
CSET (NEP) should 0N-60N, and 160W-110W (200E-250E)
ORACLES (SEA) should be 40S-5N, 20W-10E (340E-10E)

Notes on data acquisition:
for ERA5, while there exists a python API, it is not instantaneous. Either the code can be made asynchronous, or (faster) a user can download the data themselves.
SSMI: downloaded from RSS - can be automated (just wget), but then transformed with classified-code. need to add code to this repo
CERES: currently downloading from larc, using data order tool