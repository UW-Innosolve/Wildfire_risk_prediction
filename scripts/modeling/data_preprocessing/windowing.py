### OVERALL PROCESS
## pull data and labels
## determine number of lines of latitude and longitude (say 37 and 34 respectively)
## for each parameter
##      interpolate any holes # TODO ask J if there's any holes he has noticed (alternatively notice any errors later)
##      reshape each day's data to 37x34
##      create a sequence of 2D 'images' from each day (in sequential order)
## catch any outliers? # TODO add code for any days that don't follow 37x34 shape (might get caught by reshape line)
## now we have an array of sequential 2D 'image' data for each parameter
## set the windowing, say we want to use 10 days of data to train, and want to predict 5 days ahead
##      start window from day i (i>=10 because we can't start before day 11)
##      fetch window of each parameter for [i-10:i]
##      fetch window for labels for [i+5]
##      save to respective arrays for training data and labels
from idlelib.pyparse import trans

import numpy as np
import pandas as pd

## initial parameters loading
# load data (path local to Teo's machine for now)
rawdata_path = "/Users/teodoravujovic/Desktop/data/firebird/march13_pull/fb_raw_data_201407.csv"

# load raw data into pandas
rawdata_df = pd.read_csv(rawdata_path)

# get columns
# TODO fix so that the columns are not fixed
# columns_used = rawdata_df.columns[3:]
columns_used = ['10u', '10v', '2d', '2t', 'cl', 'cvh',
                'cvl', 'fal', 'lai_hv', 'lai_lv', 'lsm', 'slt', 'sp', 'src', 'stl1',
                'stl2', 'stl3', 'stl4', 'swvl1', 'swvl2', 'swvl3', 'swvl4', 'tvh',
                'tvl', 'z', 'e', 'pev', 'slhf', 'sshf', 'ssr', 'ssrd', 'str', 'strd',
                'tp', 'lightning_count', 'absv_strength_sum',
                'multiplicity_sum', 'railway_count', 'power_line_count',
                'highway_count', 'aeroway_count', 'waterway_count']
target_column = 'is_fire_day'

# count number of lines latitude and longitude (used to reshape data)
# TODO assumes that all days have same number of lat/long values
latitude_count = rawdata_df['latitude'].unique().size
longitude_count = rawdata_df['longitude'].unique().size
rows_perday = latitude_count * longitude_count

# get list of days in this file
dates = rawdata_df['date'].unique()


## create parameters and labels arrays
parameters = []
labels = rawdata_df[target_column]


## processing for each parameter
for parametername in columns_used:
    parameter_sequence = []
    parameter_full = rawdata_df[parametername]
    for day in range(len(dates)):
        parameter_ondate = parameter_full[rows_perday * day : rows_perday * (day + 1)] # get parameter values on that day
        parameter_ondate_reshaped = np.asarray(parameter_ondate).reshape(latitude_count, longitude_count) # reshape array to an 'image' according to lat/long
        parameter_sequence.append(parameter_ondate_reshaped)
    parameters.append(parameter_sequence)


def create_windows(parameters, labels, training_days, prediction_day):
    windowed_dataset = []
    windowed_labels = []

    # for every possible window between the first possible day to predict from and the last possible day to predict
    # cannot start at any value < training_days (because we don't have that data)
    # cannot predict any value > last data day + prediction_day (because we don't have that data)
    for i in range(training_days, len(parameters[1])-prediction_day):
        data_window = parameters[i-training_days:i]
        label_window = labels[i+5]
        windowed_dataset.append(data_window)
        windowed_labels.append(label_window)

    return windowed_dataset, windowed_labels

# windowed_dataset, windowed_labels = create_windows(parameters, labels, 10, 5)

print('windowing completed')
