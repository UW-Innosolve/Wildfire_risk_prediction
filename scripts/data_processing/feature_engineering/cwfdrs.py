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
      # temp = temp.pint.dequantify()
      # dew = dew.pint.dequantify()
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
    # Convert to km/h
    wind_speed = wind_speed * 3.6
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
    df['pr'] = self.raw_data['tp'] * 1000 # Convert to mm
    df['tas'] = (self.raw_data['2t'] - 273.15) + 20 # Convert to Celsius
    df['tdps'] = (self.raw_data['2d'] - 273.15) + 20 # Convert to Celsius
    
    # Compute relative humidity from dew point temperature
    hurs_rel_humidity = convert_units_to(self._rel_humidity(temp=df["tas"].to_xarray(), dew=df["tdps"].to_xarray()), "percent")
    df['hurs'] = hurs_rel_humidity
    # Compute wind speed from U and V components
    sfcWind_speed = self._wind_speed_from_components(u=df['10u'].to_xarray(), v=df['10v'].to_xarray())
    df['sfcWind'] = sfcWind_speed
    df = df.sort_values(by=['time', 'lat', 'long'])
    
    # Save the df to a CSV file for verification
    df[['time', 'lat', 'long', '10u', '10v', 'pr', 'tas', 'tdps', 'hurs', 'sfcWind']].to_csv("cwfdrs_input.csv", index=False)
    
    # Convert the DataFrame to an xarray Dataset
    ds = df.set_index(["time", "lat", "long"]).to_xarray()

    # # Sort the dataset by date to ensure the index is monotonic
    ds = ds.sortby(['time', 'lat', 'long'])

    # Assign the computed variables to the dataset
    ds = ds.assign(
        hurs=ds.hurs,  # Already in percent
        tas=ds.tas,  # Already in Celsius
        tdps=ds.tdps,  # Already in Celsius
        pr=ds.pr, # Already in mm/day
        sfcWind=ds.sfcWind, # Already in km/h
        lat=ds.lat,
        long=ds.long
    )
    # Assign all units
    ds["sfcWind"].attrs["units"] = "km h-1"
    ds["hurs"].attrs["units"] = "%"
    ds["tas"].attrs["units"] = "degC"
    ds["tdps"].attrs["units"] = "degC"
    ds["pr"].attrs["units"] = "mm/d"
    ds["lat"].attrs["units"] = "degrees_north"
    ds["long"].attrs["units"] = "degrees_east"
    
    print("Units: ", ds.tas.pint.units, ds.tdps.pint.units, ds.pr.pint.units, ds.hurs.pint.units, ds.sfcWind.pint.units)
    print("cffdrs features input data as xarray: ", ds)

    # Compute fire season mask
    season_mask = fire_season(
        tas=ds.tas,
        method="WF93",
        freq="YS",
        temp_start_thresh="12 degC",
        temp_end_thresh="5 degC",
        temp_condition_days=3,
    )

    # Compute Fire Weather Index system indices
    out_fwi = cffwis_indices(
        tas=ds.tas,
        pr=ds.pr,
        hurs=ds.hurs,
        sfcWind=ds.sfcWind,
        lat=ds.lat,
        season_mask=season_mask,
        overwintering=False,
        dry_start="CFS",
        prec_thresh="1.5 mm/d",
        dmc_dry_factor=1.2,
        carry_over_fraction=0.75,
        wetting_efficiency_fraction=0.75,
        dc_start=15,
        dmc_start=6,
        ffmc_start=85,
    )
    
    # Convert back to Pandas DataFrame
    print('------------------------------------------------------------------------------------------------------------')
    # print(out_fwi)
    self.dc = out_fwi[0].to_dataframe(name="drought_code").reset_index().sort_values(by=["time", "lat", "long"])
    self.dmc = out_fwi[1].to_dataframe(name="duff_moisture_code").reset_index().sort_values(by=["time", "lat", "long"])
    self.ffmc = out_fwi[2].to_dataframe(name="fine_fuel_moisture_code").reset_index().sort_values(by=["time", "lat", "long"])
    self.isi = out_fwi[3].to_dataframe(name="initial_spread_index").reset_index().sort_values(by=["time", "lat", "long"])
    self.bup = out_fwi[4].to_dataframe(name="build_up_index").reset_index().sort_values(by=["time", "lat", "long"])
    self.fwi = out_fwi[5].to_dataframe(name="fire_weather_index").reset_index().sort_values(by=["time", "lat", "long"])
    
    self.dc.to_csv("dc.csv", index=False)
    self.dmc.to_csv("dmc.csv", index=False)
    self.ffmc.to_csv("ffmc.csv", index=False)
    self.isi.to_csv("isi.csv", index=False)
    self.bup.to_csv("bup.csv", index=False)
    self.fwi.to_csv("fwi.csv", index=False)
    
    print('------------------------------------------------------------------------------------------------------------')

    # self.cwfdrs_features = out_fwi.to_dataframe().reset_index()
    # self.cwfdrs_features.to_csv("cwfdrs_features.csv", index=False)
      
      
  def get_features(self):
    """Return the computed CWFDRS features."""
    return self.cwfdrs_features
