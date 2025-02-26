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
    variable=['tp',
              'e',
              'sshf',      # surface_sensible_heat_flux',
              'slhf',      # surface_latent_heat_flux',
              'ssrd',      # surface_solar_radiation_downwards',
              'strd',      # surface_thermal_radiation_downwards',
              'ssr',       # surface_net_solar_radiation
              'str'],   # surface_net_thermal_radiation],
    product_type="reanalysis",
    area=[60, -120, 49, -110],  # N,W,S,E
    grid=[0.35, 0.35],
    date=['2012-05-10',
          "2012-05-11",
          "2012-05-12"],
    time=["12:00"]
    )

# lat_range=[49, 60], 
# long_range=[-120, -110],

ds.save("my_data.grib")
ds_xr = xr.open_dataset("my_data.grib", engine= "earthkit")
df = ds_xr.to_dataframe()
df.to_csv("evaporationstuff.csv")
