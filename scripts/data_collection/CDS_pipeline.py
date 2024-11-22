import cdsapi
import pandas as pd
import numpy as np
import xarray as xr  # Import xarray for working with GRIB files
from datetime import timedelta
import requests


# Initialize the CDS API client
CDS_client = cdsapi.Client(url='https://cds.climate.copernicus.eu/api', key='734d2638-ef39-4dc1-bc54-4842b788fff6')





