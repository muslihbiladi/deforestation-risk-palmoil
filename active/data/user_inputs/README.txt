palmdef-risk — User Data Input Folder
======================================
Drop your spatial files here. Set source: user in your config to use them.

Required (always):
  peatland.gpkg        — peatland polygon layer (GeoPackage or Shapefile)
  hgu.gpkg             — HGU (Hak Guna Usaha) land concession polygons
  plantation_t2.tif    — plantation raster at reference year (t2)
  plantation_t3.tif    — plantation raster at forecast year (t3, optional)

Optional overrides (set source: user in config to activate):
  protected.gpkg       — protected area polygons (overrides WDPA download)
  road.gpkg            — road network lines (overrides OSM download)
  river.gpkg           — river/waterway lines (overrides OSM download)
  town.gpkg            — settlement points as vector (overrides OSM/GHSL)
  town.tif             — settlement extent as raster (overrides OSM/GHSL)
  mill_t2.gpkg         — palm oil mill locations (overrides Trase/GFW)

File format requirements:
  Vector: GeoPackage (.gpkg) or Shapefile (.shp), must have CRS defined.
  Raster: GeoTIFF (.tif), will be reprojected to match study area CRS.
  Mill files should have a 'capacity_tonnes_ffb_hour' column for SFCA.

Default config paths already point here — just flip source: user and drop
the file. No path editing needed.
