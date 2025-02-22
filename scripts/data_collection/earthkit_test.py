import earthkit.data
from collection_utils.raw_data_assembly import RawDataAssembler
import pandas as pd
import xarray as xr
import numpy as np



ds = earthkit.data.from_source(
    "cds",
    "reanalysis-era5-single-levels",
    variable=["2t", "msl", "tvh"],
    product_type="reanalysis",
    area=[50, -10, 40, 10],  # N,W,S,E
    grid=[0.5, 0.5],
    date=["2012-05-10",
          "2012-05-11",
          "2012-05-12",
          "2012-05-13",
          "2012-05-14",
          "2012-05-15",
          "2012-05-16",
          "2012-05-17",
          "2012-05-18",
          "2012-05-19",
          "2012-05-20"],
    time=["00:00"])

ds.save("my_data.grib")
ds_xr = xr.open_dataset("my_data.grib", engine= "earthkit")
df = ds_xr.to_dataframe()
df.to_csv("multi-day-test.csv")
