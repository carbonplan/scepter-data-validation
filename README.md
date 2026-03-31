## Validation data for SCEPTER modeling


## Install

```bash
uv sync --all-groups
```


### Output data

```python
import geopandas as gpd 

gdf = gpd.read_parquet('s3://carbonplan-carbon-removal/ew-workflows-data/valdation-data/sampled_locations.parquet')
gdf
```

### Data sources

#### ISRIC

ISRIC data was pulled from a single zipped archive at: https://data.isric.org/geonetwork/srv/eng/catalog.search#/metadata/82f3d6b0-a045-4fe2-b960-6d05bc1f37c0

#### ERA5
ERA5 data was access from ECMWF. year = 2000, all months, soil vars.

#### Misc
All other data was accessible directly over the web with Xarray