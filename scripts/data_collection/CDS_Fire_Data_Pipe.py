import cdsapi
import pandas as pd
import numpy as np
import xarray as xr  # Import xarray for working with GRIB files
from datetime import timedelta
import requests

# Initialize the CDS API client
client = cdsapi.Client(url='https://cds.climate.copernicus.eu/api', key='734d2638-ef39-4dc1-bc54-4842b788fff6')

# Load wildfire data (ie. wildfire incidence data)
wildfire_data = pd.read_excel("scripts/data_collection/fp-historical-wildfire-data-2006-2023.xlsx")

# Convert 'fire_start_date' to datetime format and extract only the date part
wildfire_data['fire_start_date'] = pd.to_datetime(wildfire_data['fire_start_date'], errors='coerce')

# Filter the fire dates data to only include the relevant columns and remove rows with missing values
fire_dates = wildfire_data[['fire_start_date', 'fire_location_latitude', 'fire_location_longitude']].dropna()

# Create a DataFrame that contains every 4th day from 2006 to 2023
all_dates = pd.date_range(start="2006-01-01", end="2023-12-31", freq='4D').normalize()
all_dates = pd.Series(list(set(all_dates).union(fire_dates['fire_start_date']))).sort_values()

# Ensure all dates are within the range 2006-2023
all_dates = all_dates[(all_dates >= pd.Timestamp("2006-01-01")) & (all_dates <= pd.Timestamp("2023-12-31"))]

# Create DataFrame for all dates without fire day labels (labeling will be done later)
all_dates_df = pd.DataFrame({'date': all_dates})

# Grid of lat/long for Alberta
grid_resolution = 0.5 # 0.5 degree resolution, approximately 55km x 55km
area = [60, -120, 49, -110]

# Function to fetch weather data
#   Only fetches data for the specified variables and date range
#   See complete list of variables at: https://cds.climate.copernicus.eu/datasets/derived-era5-land-daily-statistics?tab=overview
def fetch_weather_data(start_date, end_date, variables, target_file):
    request = {
        'format': 'grib',
        'variable': variables,
        'year': list(set([str(date.year) for date in pd.date_range(start=start_date, end=end_date)])),
        'month': list(set([str(date.month).zfill(2) for date in pd.date_range(start=start_date, end=end_date)])),
        'day': list(set([str(date.day).zfill(2) for date in pd.date_range(start=start_date, end=end_date)])),
        'time': '12:00',
        'area': area,
        'grid': [grid_resolution, grid_resolution],
    }
    try:
        client.retrieve('reanalysis-era5-land', request, target_file)
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        return

# Variables to request from CDS API
variables = [
            # Temperature and pressure
            '2m_temperature', 
            'surface_pressure',
            # Wind
            '10m_u_component_of_wind', 
            '10m_v_component_of_wind',
            # Water variables
            '2m_dewpoint_temperature', 
            'total_precipitation',
            'total_evaporation',
            # Leaf area index (vegetation)
            'leaf_area_index_low_vegetation',
            'leaf_area_index_high_vegetation',
            # Heat variables (NOTE: needs review and/or reduction)
            'surface_sensible_heat_flux',
            'surface_latent_heat_flux',
            'surface_solar_radiation_downwards',
            'surface_thermal_radiation_downwards',
            'surface_net_solar_radiation',
            'surface_net_thermal_radiation',
             ]

# Function to convert GRIB to CSV
def read_grib_to_dataframe(grib_file):
    ds = xr.open_dataset(grib_file, engine='cfgrib')
    df = ds.to_dataframe().reset_index()
    df['date'] = pd.to_datetime(df['time']).dt.normalize()  # Extract only date part
    df = df.drop(columns=['number'], errors='ignore')  # Drop 'number' column if it exists
    return df

# Processing and grouping by month
grouped = all_dates_df.groupby(all_dates_df['date'].dt.to_period('M'))  # Group by month

# Adjust the tolerance for latitude and longitude proximity
latitude_tolerance = 1.0  # Increased tolerance to 1 degree
longitude_tolerance = 1.0

# Ensure fire_start_date is of type datetime.date for matching purposes
fire_dates['fire_start_date'] = fire_dates['fire_start_date'].dt.date

for period, batch in grouped:
    start_date = batch['date'].min()
    end_date = batch['date'].max()
    
    # Fetch weather data for the month
    target_file = f"weather_data_{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}.grib"
    fetch_weather_data(start_date, end_date, variables, target_file)

    # Convert GRIB to DataFrame
    weather_df = read_grib_to_dataframe(target_file)

    # Filter weather data to ensure it's within the correct date range
    weather_df = weather_df[(weather_df['date'] >= pd.Timestamp("2006-01-01")) & (weather_df['date'] <= pd.Timestamp("2023-12-31"))]

    # Convert weather_df 'date' to datetime.date type for matching purposes
    weather_df['date'] = weather_df['date'].dt.date

    # Debug: Print out the weather_df shape
    print(f"Processing weather data from {start_date} to {end_date}, Data shape: {weather_df.shape}")

    # Label fire days for the current batch by matching both date and location with a proximity check
    def is_fire_day(row):
        matching_fires = fire_dates[
            (fire_dates['fire_start_date'] == row['date']) &
            (fire_dates['fire_location_latitude'].between(row['latitude'] - latitude_tolerance,
                                                          row['latitude'] + latitude_tolerance)) &
            (fire_dates['fire_location_longitude'].between(row['longitude'] - longitude_tolerance,
                                                           row['longitude'] + longitude_tolerance))
        ]
        
        if matching_fires.empty:
            print(f"No fire match found for date {row['date']} and location ({row['latitude']}, {row['longitude']})")
        else:
            print(f"Fire match found for date {row['date']} and location ({row['latitude']}, {row['longitude']})")

        return int(not matching_fires.empty)

    # Apply the labeling function to the DataFrame
    weather_df['is_fire_day'] = weather_df.apply(is_fire_day, axis=1)

    # Check how many fire days were found
    num_fire_days = weather_df['is_fire_day'].sum()
    print(f"Number of fire days found in this batch: {num_fire_days}")

    # Save the DataFrame to a CSV file, labeled by month
    csv_output_file = f"weather_data_{period.strftime('%Y%m')}.csv"
    weather_df.to_csv(csv_output_file, index=False)
