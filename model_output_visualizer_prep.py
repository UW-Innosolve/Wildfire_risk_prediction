import numpy as np
import pandas as pd
import os

def vis_csv_maker(list_of_dates, list_of_arrays, lat_range, long_range, grid_resolution):
  # Generate latitude and longitude ranges based on the grid resolution
  latitudes = np.arange(lat_range[0], lat_range[1], grid_resolution)
  longitudes = np.arange(long_range[0], long_range[1], grid_resolution)
  
  # Create a meshgrid for latitude and longitude
  lat_grid, long_grid = np.meshgrid(latitudes, longitudes)
  
  # Flatten the grids for easier DataFrame creation
  lat_flat = lat_grid.flatten()
  long_flat = long_grid.flatten()
  
  # Initialize an empty DataFrame to store the results
  result_df = pd.DataFrame(columns=["latitude", "longitude", "risk_score", "date"])
  
  # Iterate through each date and corresponding array
  for date, array in zip(list_of_dates, list_of_arrays):
    # Flatten the array to match the lat/long grid
    risk_scores = array.flatten()
    
    # Create a temporary DataFrame for the current date
    temp_df = pd.DataFrame({
      "latitude": lat_flat,
      "longitude": long_flat,
      "risk_score": risk_scores,
      "date": date
    })
    
    # Append the temporary DataFrame to the result DataFrame
    result_df = pd.concat([result_df, temp_df], ignore_index=True)
  
  return result_df


list_of_dates = pd.date_range(start="2022-02-25", end="2022-09-26", freq="D")
lat_range = [49, 60]
long_range = [-120, -110]
grid_resolution = 0.30
output_df = vis_csv_maker(list_of_dates, list_of_arrays, lat_range, long_range, grid_resolution)
  