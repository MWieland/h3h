Changelog
=========

[vX.X.X] (20XX-XX-XX)
----------------------
Added
*******

Changed
*******

Fixed
*******

[v0.0.3] (2024-09-09)
----------------------
Added
*******
- preparation script for flood water masks
- stac catalog script to query proc-hr-semseg water masks
- optimization option in hotspots module to find best pooling weights based on kullback leibler divergence
- blob removal for water prepare script

Changed
*******
- read geopackages with bbox option set to area of interest
- improved prepare_water.py helper script
- plotting residuals instead of absolute difference in comparison
- always normalize hotspots with quantiles [0,1] instead of user defined
- updated testdata
- renamed optimize module into gridsearch

Fixed
*******
- point conversion also requires an attribute field named "value"

[v0.0.2] (2024-01-24)
----------------------
Added
*******
- test data
- area of interest for hotspot computation
- save raw hotspot values
- additional regression metrics
- notebook to explore results of optimize module

Changed
*******
- optimization module
- regression with statsmodels

Fixed
*******
- compute hotspots for an entire aoi and not just the intersect of input layers
- compare hotspots only makes sense when computed over the same aoi
- normalize input layers after joining them to aoi
- statsmodels regression requires to add constant

[v0.0.1]  (2024-01-16)
----------------------
- initial release
