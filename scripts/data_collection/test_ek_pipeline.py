import earthkit.data
from collection_utils.raw_data_assembly import RawDataAssembler
import pandas as pd
import xarray as xr
import numpy as np
import os

# Set the API key
os.environ['CDSAPI_KEY'] = "734d2638-ef39-4dc1-bc54-4842b788fff6"

ds = earthkit.data.from_source(
    "cds",
    "reanalysis-era5-single-levels",
    variable=[  # Temperature and pressure
                 '2t',      # 2m_temperature 
                 'sp',       # surface_pressure
                 # Wind
                 '10u',      # 10m_u_component_of_wind', 
                 '10v',      # 10m_v_component_of_wind',
                 # Water variables
                #  '2m_dewpoint_temperature',      # 2m_dewpoint_temperature', 
                # NOTE: precipitation accumulations need to be repaired
                  'tp',       # total_precipitation',
                  'e',        # total_evaporation',
                # Leaf area index (vegetation)
                 'lai_lv',   # leaf_area_index_low_vegetation',
                 'lai_hv'   # leaf_area_index_high_vegetation',
                # # Heat variables (NOTE: needs to be repaired, if the values are useful)
                 'sshf',      # surface_sensible_heat_flux',
                 'slhf',      # surface_latent_heat_flux',
                 'ssrd',      # surface_solar_radiation_downwards',
                 'strd',      # surface_thermal_radiation_downwards',
                 'ssr',       # surface_net_solar_radiation
                 'str',       # surface_net_thermal_radiation
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
    date=['2012-05-10',
          "2012-05-11",
          "2012-05-12"],
    # time=["12:00"]
    )

# lat_range=[49, 60], 
# long_range=[-120, -110],

ds.save("my_data.grib")
ds_xr = xr.open_dataset("my_data.grib", engine= "earthkit")
df = ds_xr.to_dataframe()
df.to_csv("evaporationstuff.csv")
