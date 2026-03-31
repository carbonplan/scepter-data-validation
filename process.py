import geopandas as gpd 
import rioxarray 
import xarray as xr 
import pandas as pd 
import obstore
from virtualizarr.registry import ObjectStoreRegistry
from virtual_tiff import VirtualTIFF
import rasterix
from affine import Affine


PROTOCOL = "s3://"
BUCKET = "carbonplan-carbon-removal/"
VALIDATION_DATA_PREFIX = "ew-workflows-data/valdation-data"

def fetch_scepter_locations() ->gpd.GeoDataFrame:
    loc_path = 's3://carbonplan-carbon-removal/ew-workflows-data/gcam/apprate_calculated/one-cell-per-region/exe_limBio1p5/annual_rockflx_densest-ag-region_1deg.pkl'
    return pd.read_pickle(loc_path).drop_duplicates(subset=["lat", "lon"]).reset_index().drop(['index'],axis=1)

def ERA5()->xr.Dataset:
    ds = xr.open_dataset(f'{PROTOCOL}{BUCKET}{VALIDATION_DATA_PREFIX}/ERA5/data_stream-moda_stepType-avgua.nc', engine='h5netcdf', chunks="auto")
    ds = ds.rename({'longitude':'lon','latitude':'lat'})
    ds = ds.drop(['expver','number'])
    ds.coords['lon'] = (ds.coords['lon'] + 180) % 360 - 180
    ds = ds.sortby(ds['lon'])
    
    return ds['stl1'].mean(dim='valid_time')

def GLDAS()->xr.Dataset:
    return xr.open_dataset('https://ldas.gsfc.nasa.gov/sites/default/files/ldas/gldas/SOILS/GLDASp5_porosity_025d.nc4', engine='h5netcdf', chunks="auto")

ISRIC_VARS = ["BSAT", "CECS", "CECc", "ECEC", "PHAQ", "ORGC"]
ISRIC_DEPTHS = ["D1", "D2", "D3", "D4", "D5", "D6", "D7"]
ISRIC_DEPTH_LABELS = ["0-20cm", "20-40cm", "40-60cm", "60-80cm", "80-100cm", "100-150cm", "150-200cm"]

def ISRIC_raster() -> xr.DataArray:
    return xr.open_dataset(
        f'{PROTOCOL}{BUCKET}{VALIDATION_DATA_PREFIX}/ISRIC/wise_30sec_v1.tif',
        engine='rasterio', chunks="auto"
    )['band_data'].squeeze('band')

def ISRIC_tables() -> tuple[pd.Series, dict]:
    """We need the lookup tables for location the vars. Read our TSV"""
    tsv = pd.read_csv(
        f'{PROTOCOL}{BUCKET}{VALIDATION_DATA_PREFIX}/ISRIC/wise_30sec_v1.tsv', sep='\t'
    ).rename(columns={'pixel_vaue': 'pixel_value'})
    pixel_to_newsuid = tsv.set_index('pixel_value')['description']

    depth_tables = {}
    for d, label in zip(ISRIC_DEPTHS, ISRIC_DEPTH_LABELS):
        df = pd.read_csv(
            f'{PROTOCOL}{BUCKET}{VALIDATION_DATA_PREFIX}/ISRIC/HW30s_w{d}.txt', quotechar='"'
        )
        depth_tables[label] = df.set_index('NEWSUID')[ISRIC_VARS]

    return pixel_to_newsuid, depth_tables


def MODIS(MODIS_YEAR: int = 2000) ->xr.Dataset:
    return xr.open_dataset(f'http://files.ntsg.umt.edu/data/NTSG_Products/MOD17/GeoTIFF/MOD17A3/GeoTIFF_30arcsec/MOD17A3_Science_NPP_{MODIS_YEAR}.tif', engine='rasterio', chunks="auto")


def SOILGRIDS()->xr.Dataset:
    bucket_url = "https://files.isric.org/"
    s3_store = obstore.store.from_url(bucket_url)
    registry = ObjectStoreRegistry({bucket_url: s3_store})
    parser = VirtualTIFF(ifd=0)

    variables = ['phh2o', 'soc', 'bdod', 'cec']
    depths = ['0-5cm', '5-15cm', '15-30cm', '30-60cm', '60-100cm', '100-200cm']

    def _open_manifest(var, depth, res=5000):
        url = f"{bucket_url}soilgrids/latest/data_aggregated/{res}m/{var}/{var}_{depth}_mean_{res}.tif"
        manifest_store = parser(url=url, registry=registry)
        da = xr.open_zarr(manifest_store, zarr_format=3, consolidated=False)['0']
        return da

    da_ref = _open_manifest('phh2o', '0-5cm')
    model_pixel_scale = da_ref.attrs['model_pixel_scale']
    model_tiepoint = da_ref.attrs['model_tiepoint']
    transform = Affine(
        model_pixel_scale[0], 0.0, model_tiepoint[3],
        0.0, -model_pixel_scale[1], model_tiepoint[4]
    )

    depth_slices = []
    for depth in depths:
        slice_das = []
        for var in variables:
            da = _open_manifest(var, depth)
            da = da.proj.assign_crs(spatial_ref="ESRI:54052")
            index = rasterix.RasterIndex.from_transform(transform, width=da.sizes['x'], height=da.sizes['y'])
            da = da.assign_coords(xr.Coordinates.from_xindex(index))
            slice_das.append(da.rename(var))
        depth_slices.append(xr.merge(slice_das))

    combined = xr.concat(depth_slices, dim=pd.Index(depths, name='depth'))
    combined_crs = combined.rio.write_crs("ESRI:54052")
    return combined_crs.rio.reproject("EPSG:4326")


def main() -> gpd.GeoDataFrame:
    df = fetch_scepter_locations()

    era5_da = ERA5()
    era5_sampled = era5_da.sel(
        lon=xr.DataArray(df.lon.values, dims='points'),
        lat=xr.DataArray(df.lat.values, dims='points'),
        method='nearest',
    ).compute().values

    gldas_ds = GLDAS()
    gldas_sampled = gldas_ds['GLDAS_porosity'].squeeze('time').sel(
        lon=xr.DataArray(df.lon.values, dims='points'),
        lat=xr.DataArray(df.lat.values, dims='points'),
        method='nearest',
    ).compute().values

    # ISRIC — sample raster for SMU codes, then join property tables by depth
    isric_raster = ISRIC_raster()
    pixel_to_newsuid, depth_tables = ISRIC_tables()
    pixel_vals = isric_raster.sel(
        x=xr.DataArray(df.lon.values, dims='points'),
        y=xr.DataArray(df.lat.values, dims='points'),
        method='nearest',
    ).compute().values.astype(int)
    newsuids = pixel_to_newsuid.reindex(pixel_vals).values

    modis_ds = MODIS()
    modis_sampled = modis_ds['band_data'].isel(band=0).sel(
        x=xr.DataArray(df.lon.values, dims='points'),
        y=xr.DataArray(df.lat.values, dims='points'),
        method='nearest',
    ).compute().values

    # SoilGrids — multi-variable × depth; unstack depth into flat columns
    soilgrids_ds = SOILGRIDS()
    soilgrids_sampled = soilgrids_ds.sel(
        x=xr.DataArray(df.lon.values, dims='points'),
        y=xr.DataArray(df.lat.values, dims='points'),
        method='nearest',
    )
    soilgrids_df = (
        soilgrids_sampled.compute()
        .drop_vars(['x', 'y', 'spatial_ref'])
        .to_dataframe()
        .unstack('depth')
    )
    soilgrids_df.columns = [f"SoilGrids_{var}_{depth}" for var, depth in soilgrids_df.columns]
    soilgrids_df = soilgrids_df.reset_index(drop=True)

    isric_df = pd.DataFrame(index=range(len(df)))
    for depth_label, props_df in depth_tables.items():
        joined = props_df.reindex(newsuids).reset_index(drop=True)
        for var in ISRIC_VARS:
            isric_df[f'ISRIC_{var}_{depth_label}'] = joined[var].values

    result = df.copy()
    result['ERA5_stl1'] = era5_sampled
    result['soil_porosity_GLDAS'] = gldas_sampled
    result['MODIS_NPP'] = modis_sampled
    result = pd.concat([result, soilgrids_df, isric_df], axis=1)

    gdf = gpd.GeoDataFrame(
        result,
        geometry=gpd.points_from_xy(df.lon, df.lat),
        crs="EPSG:4326",
    ).drop(columns=['lat', 'lon'])
    gdf.to_parquet(f'{PROTOCOL}{BUCKET}{VALIDATION_DATA_PREFIX}/sampled_locations.parquet')


if __name__ == "__main__":
    gdf = main()