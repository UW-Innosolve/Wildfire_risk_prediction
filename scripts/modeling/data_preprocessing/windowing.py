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
import torch


def reshape_data(df, features, target_column, device):
    # count number of lines latitude and longitude (used to reshape data)
    # TODO assumes that all days have same number of lat/long values
    # TODO update to only BUT KEEP INDEX
    latitude_count = df['latitude'].unique().size
    longitude_count = df['longitude'].unique().size
    rows_perday = latitude_count * longitude_count

    # get list of days in this file
    dates = df['date'].unique()
    print(f"Number of unique dates in dataset is {len(dates)}")

    # create parameters and labels arrays
    parameters = []
    labels = []

    # create corresponding labels for each day
    label_full = df[target_column]

    for day in range(len(dates) - 1):
        labels_ondate = label_full[rows_perday * day: rows_perday * (day + 1)]
        labels_ondate_reshaped = np.asarray(labels_ondate).reshape(latitude_count, longitude_count)
        labels.append(labels_ondate_reshaped)

        parameter_sequence = []
        for parametername in features:
            print(parametername)
            parameter_full = df[parametername]

            parameter_ondate = parameter_full[rows_perday * day: rows_perday * (day + 1)]  # get parameter values on that day
            parameter_ondate_reshaped = np.asarray(parameter_ondate).reshape(latitude_count,longitude_count)  # reshape array to an 'image' according to lat/long
            parameter_sequence.append(parameter_ondate_reshaped)

        parameters.append(parameter_sequence)

    # # create corresponding labels for each day
    # label_full = torch.tensor(df[target_column].array)#, device=device)
    #
    # for day in range(len(dates) - 1):
    #     labels_ondate = label_full[rows_perday * day: rows_perday * (day + 1)]
    #     labels_ondate_reshaped = labels_ondate.reshape(latitude_count, longitude_count)
    #     labels.append(labels_ondate_reshaped)
    #
    #     parameter_sequence = []
    #     for parametername in features:
    #         print(parametername)
    #         parameter_full = torch.tensor(df[parametername].array)#, device=device)
    #
    #         parameter_ondate = parameter_full[rows_perday * day: rows_perday * (day + 1)]  # get parameter values on that day
    #         parameter_ondate_reshaped = parameter_ondate.reshape(latitude_count,longitude_count)  # reshape array to an 'image' according to lat/long
    #         parameter_sequence.append(parameter_ondate_reshaped)
    #
    #     parameters.append(parameter_sequence)

    # return torch.tensor(parameters, device=device), torch.tensor(labels, device=device)
    return np.asarray(parameters), np.asarray(labels)


def create_windows(parameters, labels, training_days, prediction_day):
    windowed_dataset = []
    windowed_labels = []

    # for every possible window between the first possible day to predict from and the last possible day to predict
    # cannot start at any value < training_days (because we don't have that data)
    # cannot predict any value > last data day + prediction_day (because we don't have that data)
    for i in range(training_days, len(parameters[1])-prediction_day):
        print(i)
        print(f'Training on days {i-training_days} through day {i}, predicting day {i+prediction_day}')
        data_window = parameters[:, i-training_days:i, :, :]
        label_window = labels[i+prediction_day, :, :]
        windowed_dataset.append(data_window)
        windowed_labels.append(label_window)

    return windowed_dataset, windowed_labels


def create_indexed_windows(indexed_day, parameters, labels, training_days, prediction_day):
    windowed_data = parameters[:, indexed_day - training_days:indexed_day, :, :]
    windowed_label = labels[indexed_day + prediction_day, :, :]

    return windowed_data, windowed_label


def reshape_and_window_indexed(indexed_day, raw_dataframe, labels, training_days, prediction_day):
    pass


def batched_indexed_windows(batch_indices, parameters_full, labels_full, training_days, prediction_day, device='cpu'):
    with torch.no_grad():
        batch_data_windows = parameters_full[batch_indices[0] - training_days:batch_indices[0], :, :].unsqueeze(0)
        batch_label_windows = labels_full[batch_indices[0] + prediction_day, :, :].unsqueeze(0)

        for indx in batch_indices[1:]:
            windowed_data = parameters_full[indx - training_days:indx, :, :].unsqueeze(0)
            batch_data_windows = torch.cat((batch_data_windows, windowed_data), dim=0)
            windowed_label = labels_full[indx + prediction_day, :, :].unsqueeze(0)
            batch_label_windows = torch.cat((batch_label_windows, windowed_label), dim=0)

    return batch_data_windows, batch_label_windows

## initial parameters loading
# load data (path local to Teo's machine for now)
# rawdata_path = "/Users/teodoravujovic/Desktop/data/firebird/march13_pull/fb_raw_data_201407.csv"
#
# # load raw data into pandas
# rawdata_df = pd.read_csv(rawdata_path)
#
# # get columns
# # TODO fix so that the columns are not fixed
# # columns_used = rawdata_df.columns[3:]
# columns_used = ['10u', '10v', '2d', '2t', 'cl', 'cvh',
#                 'cvl', 'fal', 'lai_hv', 'lai_lv', 'lsm', 'slt', 'sp', 'src', 'stl1',
#                 'stl2', 'stl3', 'stl4', 'swvl1', 'swvl2', 'swvl3', 'swvl4', 'tvh',
#                 'tvl', 'z', 'e', 'pev', 'slhf', 'sshf', 'ssr', 'ssrd', 'str', 'strd',
#                 'tp', 'lightning_count', 'absv_strength_sum',
#                 'multiplicity_sum', 'railway_count', 'power_line_count',
#                 'highway_count', 'aeroway_count', 'waterway_count']
# target_column = 'is_fire_day'
#
# reshaped_data, reshaped_labels = reshape_data(rawdata_df, columns_used, target_column)
# windowed_dataset, windowed_labels = create_windows(reshaped_data, reshaped_labels, 10, 5)
# window_daytwelve, window_label_daytwelve = create_indexed_windows(12, reshaped_data, reshaped_labels, 10, 5)
#
# print('windowing completed')
