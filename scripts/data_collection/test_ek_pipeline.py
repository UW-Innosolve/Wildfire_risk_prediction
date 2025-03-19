import earthkit.data
from collection_utils.raw_data_assembly import RawDataAssembler
import pandas as pd
import xarray as xr
import numpy as np
import os

# Set the API key
os.environ['CDSAPI_KEY'] = "734d2638-ef39-4dc1-bc54-4842b788fff6"

#TODO: Add actual tests for the pipeline, one for each parameter type (variant, invariant, and accumulated)

ds = earthkit.data.from_source(
    "cds",
    "reanalysis-era5-single-levels",
    variable=[  # Temperature and pressure
                 '2m_temperature',      # 2m_temperature 
                 'sp',       # surface_pressure
                 # Wind
                 '10u',      # 10m_u_component_of_wind', 
                 '10v',      # 10m_v_component_of_wind',
                 # Water variables
                 '2m_dewpoint_temperature',      # 2m_dewpoint_temperature', 
                # NOTE: precipitation accumulations need to be repaired
                #   'tp',       # total_precipitation',
                #   'e',        # total_evaporation',
                # Leaf area index (vegetation)
                 'lai_lv',   # leaf_area_index_low_vegetation',
                 'lai_hv'   # leaf_area_index_high_vegetation',
                # # Heat variables (NOTE: needs to be repaired, if the values are useful)
                #  'sshf',      # surface_sensible_heat_flux',
                #  'slhf',      # surface_latent_heat_flux',
                #  'ssrd',      # surface_solar_radiation_downwards',
                #  'strd',      # surface_thermal_radiation_downwards',
                #  'ssr',       # surface_net_solar_radiation
                #  'str',       # surface_net_thermal_radiation
                 'tvl', # low_veg_cover
                 'tvh', # high_veg_cover
                 'cvl', # low_veg_type
                 'cvh',  # high_veg_type
                 # Lakes and rivers
                 'cl',  # lake_cover
                 'lsm', # land_sea_mask
                 # Topography
                 'z'    # Geopotential (proportional to elevation, not linearly due to oblong shape of Earth)
                 ],      # surface_net_thermal_radiation],
    product_type="reanalysis",
    area=[60, -120, 49, -110],  # N,W,S,E
    grid=[0.35, 0.35],
    date=['2006-08-01',
          '2006-08-02',
          '2006-08-03',
          '2006-08-04',
          '2006-08-05',
          '2006-08-06',
          '2006-08-07',
          '2006-08-08',
          '2006-08-09',
          '2006-08-10',
          '2006-08-11',
          '2006-08-12',
          '2006-08-13',
          '2006-08-14',
          '2006-08-15',
          '2006-08-16',
          '2006-08-17',
          '2006-08-18',
          '2006-08-19',
          '2006-08-20',
          '2006-08-21',
          '2006-08-22',
          '2006-08-23',
          '2006-08-24',
          '2006-08-25',
          '2006-08-26',
          '2006-08-27',
          '2006-08-28',
          '2006-08-29',
          '2006-08-30',
          '2006-08-31'],
    time=["15:00"]
    )

# lat_range=[49, 60], 
# long_range=[-120, -110],

ds.save("aug_test.grib")
ds_xr = xr.open_dataset("aug_test.grib", engine= "earthkit")
df = ds_xr.to_dataframe()
df.to_csv("aug_test.csv")
