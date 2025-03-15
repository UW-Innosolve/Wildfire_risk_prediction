from xclim.indices.fire import cffwis_indices
import xclim.indices as xci
import xarray as xr
import numpy as np
import pandas as pd
import xclim
import pint_xarray
from xclim.indices import cffwis_indices, fire_season
from xclim.core.units import convert_units_to


class FbCwfdrsFeatures():
  def __init__(self, raw_data_df):
    """
    Initialize the class with raw input data (Pandas DataFrame).
    Convert it to an xarray.Dataset for processing.
    """
    self.raw_data = raw_data_df
    self.cwfdrs_inputs = self._convert_to_xarray(raw_data_df)
    self.cwfdrs_features = pd.DataFrame()
    
    self.config_features()
    self.compute_cwfdrs()

  def _convert_to_xarray(self, df):
    """
    Convert the Pandas DataFrame into an xarray.Dataset.
    Assumes 'date', 'latitude', and 'longitude' columns exist.
    """
    return df.set_index(["date", "latitude", "longitude"]).to_xarray()

  def config_features(self):
    """Define the feature names for the dataset."""
    self.cwfdrs_features = ['drought_code',
                            'duff_moisture_code',
                            'fine_fuel_moisture_code',
                            'initial_spread_index',
                            'build_up_index',
                            'fire_weather_index']
  
  def _rel_humidity(self, temp, dew):
      """
      Compute relative humidity from temperature and dew point temperature.
      Returns an xarray DataArray instead of a Dataset.
      """
      temp = temp.pint.dequantify()
      dew = dew.pint.dequantify()
      e_t = np.exp((17.625 * temp) / (243.04 + temp))
      e_d = np.exp((17.625 * dew) / (243.04 + dew))
      rel_humidity = 100 * (e_d / e_t)
      return xr.DataArray(rel_humidity, dims=temp.dims, coords=temp.coords, attrs={"units": "%"})
    
  def _wind_speed_from_components(self, u, v):
    """
    Compute wind speed from U and V components.
    """
    u = u.pint.dequantify()
    v = v.pint.dequantify()
    wind_speed = np.sqrt(u ** 2 + v ** 2)
    return xr.DataArray(wind_speed, dims=u.dims, coords=u.coords, attrs={"units": "m s-1"})


  def compute_cwfdrs(self):
    """
    Calculate the Canadian Fire Weather Danger Rating System (CWFDRS) indices.
    """

    # Load your DataFrame
    df = pd.DataFrame()
    df['time'] = self.raw_data['date']
    df['lat'] = self.raw_data['latitude']
    df['long'] = self.raw_data['longitude']
    df['10u'] = self.raw_data['10u']
    df['10v'] = self.raw_data['10v']
    df['tp'] = self.raw_data['tp'] / 1000 # Convert to mm
    df['tas'] = self.raw_data['2t'] - 273.15 # Convert to Celsius
    df['tdps'] = self.raw_data['2d'] - 273.15 # Convert to Celsius
    
    # Convert to xarray Dataset
    ds = df.set_index(['time', 'lat', 'long']).to_xarray()
    
    # Sort the dataset by date to ensure the index is monotonic
    ds = ds.sortby('time')

    # Compute relative humidity from dew point temperature
    hurs_rel_humidity = convert_units_to(self._rel_humidity(temp=ds.tas, dew=ds.tdps), "percent")
    
    # Compute wind speed from U and V components
    sfcWind_speed = self._wind_speed_from_components(u=ds['10u'], v=ds['10v'])

    # Assign the computed variables to the dataset
    ds = ds.assign(
        hurs=hurs_rel_humidity,
        tas=ds.tas,  # Already in Celsius
        tdps=ds.tdps,  # Already in Celsius
        pr=ds.tp, # Already in mm/day
        sfcWind=sfcWind_speed,
        lat=ds.lat,
        long=ds.long
    )
    
    # Assign all units
    ds["sfcWind"].attrs["units"] = "m s-1"
    ds["hurs"].attrs["units"] = "%"
    ds["tas"].attrs["units"] = "degC"
    ds["tdps"].attrs["units"] = "degC"
    ds["pr"].attrs["units"] = "mm/d"
    ds["lat"].attrs["units"] = "degrees_north"
    ds["long"].attrs["units"] = "degrees_east"
    
    print("Units: ", ds.tas.pint.units, ds.tdps.pint.units, ds.pr.pint.units, ds.hurs.pint.units, ds.sfcWind.pint.units)
    print("Data: ", ds)

    # Compute fire season mask
    season_mask = fire_season(
        tas=ds.tas,
        method="WF93",
        freq="YS",
        temp_start_thresh="12 degC",
        temp_end_thresh="5 degC",
        temp_condition_days=3,
    )
    
    print(season_mask.dims)

    # Compute Fire Weather Index system indices
    out_fwi = cffwis_indices(
        tas=ds.tas,
        pr=ds.pr,
        hurs=ds.hurs,
        sfcWind=ds.sfcWind,
        lat=ds.lat,
        season_mask=season_mask,
        overwintering=True,
        dry_start="CFS",
        prec_thresh="1.5 mm/d",
        dmc_dry_factor=1.2,
        carry_over_fraction=0.75,
        wetting_efficiency_fraction=0.75,
        dc_start=15,
        dmc_start=6,
        ffmc_start=85,
    )



    # # self.cwfdrs_inputs["sfcWind"] = xci.uas_vas_2_sfcwind(
    # # uas=self.raw_data["10u"].to_xarray(), 
    # # vas=self.raw_data["10v"].to_xarray()
    # # )

    # # Attach units to the wind components
    # units = "m s-1"  # Assuming 10m wind components are in meters per second
    # u = self.raw_data["10u"].to_xarray().pint.quantify(units)
    # v = self.raw_data["10v"].to_xarray().pint.quantify(units)
    # # Compute wind speed
    # self.cwfdrs_inputs["sfcWind"] = xci.uas_vas_2_sfcwind(uas=u, vas=v)
    
    # # Compute wind speed from U/V vector components
    # # wind_vector = xr.Dataset({
    # #     "u": self.raw_data["10u"],
    # #     "v": self.raw_data["10v"]
    # # })
    # # self.cwfdrs_inputs["sfcWind"] = wind_vector

    # # Assign precipitation
    # self.cwfdrs_inputs["pr"] = self.raw_data["tp"]  # Precipitation in mm/day

    # # Extract latitude (assumed to be constant across dataset)
    # self.cwfdrs_inputs["lat"] = self.raw_data["latitude"]

    # # Compute Fire Weather Indices
    # fire_indices = cwfdrs.cffwis_indices(
    #     tas=self.cwfdrs_inputs["tas"],
    #     pr=self.cwfdrs_inputs["pr"],
    #     hurs=self.cwfdrs_inputs["hurs"],
    #     sfcWind=self.cwfdrs_inputs["sfcWind"],
    #     lat=self.cwfdrs_inputs["lat"],
    #     overwintering=True,
    #     dry_start="CFS"
    # )

    # Convert back to Pandas DataFrame
    print(out_fwi)
    print(type(out_fwi))
    # self.cwfdrs_features = out_fwi.to_dataframe().reset_index()
      
  def get_features(self):
    """Return the computed CWFDRS features."""
    return self.cwfdrs_features

# from xclim.indices import cwfdrs
# import xarray as xr
# import numpy as np
# import pandas as pd

# class FbCwfdrsFeatures():
#   def __init__(self, raw_data_df):
#     self.raw_data = raw_data_df
#     self.cwfdrs_inputs = xr.Dataset()
#     self.cwfdrs_features = pd.DataFrame()
    
#   def config_features(self):
#     self.cwfdrs_features = ['drought_code',
#                             'duff_moisture_code',
#                             'fine_fuel_moisture_code',
#                             'initial_spread_index',
#                             'build_up_index',
#                             'fire_weather_index',
#                             'seasonal_drought_index']
    
#   def cwfdrs(self):
#     """
#     Calculate the Canadian Fire Weather Danger Rating System (CWFDRS) indices.
#     """
#     # Convert temperature to Celsius
#     self.cwfdrs_inputs['c_temp'] = self.raw_data["2t"] - 273.15
#     self.cwfdrs_inputs['relative_humidity'] = cwfdrs.relative_humidity(
#       tas=self.cwfdrs_inputs['c_temp'],
#       tdps=self.raw_data["2d"] - 273.15
#     )
#     self.cwfdrs_inputs['wind_speed'] = cwfdrs.wind_speed(
#       u=self.raw_data["10u"],
#       v=self.raw_data["10v"]
#     )
#     self.cwfdrs_inputs['precipitation'] = self.raw_data["sf"]
#     self.cwfdrs_inputs['latitude'] = self.raw_data["latitude"]
#     self.cwfdrs_inputs['longitude'] = self.raw_data["longitude"]
    
    


# from xclim.indices import cwfdrs
# import xarray as xr
# import numpy as np
# import pandas as pd

# class FbCwfdrsFeatures():
#   def __init__(self, raw_data_df):
#     self.raw_data = raw_data_df
#     self.cwfdrs_features = pd.DataFrame()
    
#   def config_features(self):
#     self.cwfdrs_features = ['drought_code',
#                             'duff_moisture_code',
#                             'fine_fuel_moisture_code',
#                             'initial_spread_index',
#                             'build_up_index',
#                             'fire_weather_index',
#                             'seasonal_drought_index']
    
#   def cwfdrs(self):
#     """
#     Calculate the Canadian Fire Weather Danger Rating System (CWFDRS) indices.
#     """
#     # Convert temperature to Celsius
#     self.cwfdrs_features['c_temp'] = self.raw_data["2t"] - 273.15
#     pass