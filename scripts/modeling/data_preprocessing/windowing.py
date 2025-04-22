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
import logging


# Configure logging: INFO level logs progress, DEBUG could be used for more details.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')


def create_fire_region_masks(targets, tolerance):
    """
    Description: creates a mask of any areas specified as class 1 (fire) in targets and include all regions within the tolerance value.
    For each grid point of value of 1 in the target array, all horizontal and vertical grid points in the mask array +/- tolerance will also be set to 1.

    Parameters:
        targets: np.ndarray
            Groundtruth target array
        tolerance: int
            How much padding to include around fire regions.
            Applied to both horizontal and vertical directions.

    Returns:
        mask: np.ndarray
            Masked 'fire region' including tolerance
    """
    # generate blank grid for output mask
    grid = targets.shape
    mask = torch.zeros(grid)

    # generate output mask according to specifications
    for i in range(tolerance - 1, grid[0] - tolerance): # ensure we don't go past the edges
        for j in range(tolerance - 1, grid[1] - tolerance):
            if targets[i, j] == 1:
                mask[i - tolerance:i + tolerance, j - tolerance:i + tolerance] = 1

    return mask


# def create_fire_region_column(df, tolerance):


def reshape_data(df, features, target_column, device, include_masks=False, tolerance=2):
    """
    Description: Reshapes all features into a 2-dimensional latitude x longitude grid. Returns a 3 sequential arrays of reshaped parameters, labels, and masks (if applicable) for each day.

    Parameters:
        df: pandas dataframe
        features: list of strings
            Column names of features to reshape in df
        target_column: string
            Name of target column in df (used as labels)
        device: torch.device
            Device to
            Not used in this iteration as GPU implementation is left out
        include_masks: bool
            Specify whether to include masks created from corresponding labels
        tolerance: int
            Specify how much to pad fire areas in the generated masks, if applicable

    Returns:
        np.array, np.array, np.array
            Returned parameter array will have shape [number of days x number of features x latitude x longitude]
            Returned label array will have shape [number of days x latitude x longitude]
            Returned mask array will have shape [number of days x latitude x longitude]

    Notes:
        - Uses numpy instead of torch implementation as torch ran extremely slowly (even on GPU)
        - Uses row indices (rows per day) instead of dates for speed improvements (especially relevant for local execution)
        - Assumes that data is complete and not missing any rows (correct for processed dataset as of March 28, 2025)
        - Assumes all days have the same number of lat/long values
    """
    # count number of lines latitude and longitude (used to reshape data)
    latitude_count = df['latitude'].unique().size
    longitude_count = df['longitude'].unique().size
    rows_perday = latitude_count * longitude_count

    # get list of days in the dataframe
    dates = df['date'].unique()
    print(f"Number of unique dates in dataset is {len(dates)}")

    # create empty list of parameters, labels, and masks arrays
    parameters = []
    labels = []
    masks = []

    # pull complete list of labels from dataframe
    label_full = df[target_column]

    for day in range(len(dates) - 1):
        # get label values for specified day
        labels_ondate = label_full[rows_perday * day: rows_perday * (day + 1)]
        # reshape to 2D grid according to latitude and longitude
        labels_ondate_reshaped = np.asarray(labels_ondate).reshape(latitude_count, longitude_count)
        # add to complete list of labels
        labels.append(labels_ondate_reshaped)

        # create blank sequence of parameters for day
        parameter_sequence = []
        # generate reshaped 2D grid data for each parameter and append to complete list
        for parametername in features:
            # print(parametername)
            # pull complete array of
            # not optimized!
            parameter_full = df[parametername]
            # get parameter values for specified day
            parameter_ondate = parameter_full[rows_perday * day: rows_perday * (day+1)]
            # reshape to 2D grid according to latitude and longitude
            parameter_ondate_reshaped = np.asarray(parameter_ondate).reshape(latitude_count, longitude_count)
            # add to list of parameters on day
            parameter_sequence.append(parameter_ondate_reshaped)

        # add parameters on day to the complete list of parameters
        parameters.append(parameter_sequence)

    # construct full set of masks if specified
    if include_masks:
        for i in range(len(labels)):
            mask = create_fire_region_masks(labels[i], tolerance=tolerance)
            masks.append(mask)

    # return lists as np arrays
    return np.asarray(parameters), np.asarray(labels), np.asarray(masks)


def create_windows(parameters, labels, training_days, prediction_day):
    windowed_dataset = []
    windowed_labels = []

    # for every possible window between the first possible day to predict from and the last possible day to predict
    # cannot start at any value < training_days (because we don't have that data)
    # cannot predict any value > last data day + prediction_day (because we don't have that data)
    for i in range(training_days, len(parameters[1]) - prediction_day):
        print(i)
        print(f'Training on days {i - training_days} through day {i}, predicting day {i + prediction_day}')
        data_window = parameters[:, i - training_days:i, :, :]
        label_window = labels[i + prediction_day, :, :]
        windowed_dataset.append(data_window)
        windowed_labels.append(label_window)

    return windowed_dataset, windowed_labels


def create_indexed_windows(indexed_day, parameters, labels, training_days, prediction_day):
    windowed_data = parameters[:, indexed_day - training_days:indexed_day, :, :]
    windowed_label = labels[indexed_day + prediction_day, :, :]

    return windowed_data, windowed_label


def reshape_and_window_indexed(indexed_day, raw_dataframe, labels, training_days, prediction_day):
    pass


def batched_indexed_windows(batch_indices, parameters_full, labels_full, training_days, prediction_day, device='cpu',
                            include_masks=False, masks_full=None):
    """
    Description: Given a set of indices, creates a window of length training_days and their corresponding label for each index
    Returns array of windowed data with shape [batch_size x training_days x num_features x latitude x longitude] and corresponding array of labels
    Also includes windowed array of masks for each index if specified

    Parameters:
        batch_indices: list of ints
            indices of each sample in the batch
            each index corresponds to a particular day in the dataset
        parameters_full: torch.Tensor
            complete list of reshaped parameters for entire dataset
        labels_full: torch.Tensor
            complete list of reshaped labels for entire dataset
        training_days: int
            set how many sequential days to pull for each sample as input data
            training days for each index in batch_indices will include from day index - training_days through day index - 1
        prediction_day: int
            set prediction horizon
            predicted days will be day index + prediction_day for each index in batch_indices
        device: torch.device
            Default: 'cpu'
            specify which device is being used for computation
            unused for now as everything is already assigned a device
        include_masks: bool
            Default: False
            specify whether to include masks
        masks_full: None or torch.Tensor
            Default: None
            complete list of reshaped masks for entire dataset (if applicable)

    Returns:
        [torch.Tensor, torch.Tensor, torch.Tensor]
            tensors for batch of input parameters, labels, and masks corresponding to the indices in batch_indices

    Notes:
        - Each day is assigned an index from 1 through 6570 corresponding to the date between Jan 1, 2006 and Dec 31, 2024
        - Uses indices (not dates directly) to determine window start / end points in data
        - February 29th in leap years is skipped!! (due to a pandas datetime problem)
    """
    with torch.no_grad(): # ensure gradients of tensors are not affected
        # create initial tensors for all arrays using the first index in the batch
        batch_data_windows = parameters_full[batch_indices[0] - training_days:batch_indices[0], :, :].unsqueeze(0)
        batch_label_windows = labels_full[batch_indices[0] + prediction_day, :, :].unsqueeze(0)
        if include_masks:
            batch_mask_windows = masks_full[batch_indices[0] + prediction_day, :, :].unsqueeze(0)

        # generate array of corresponding input parameters and labels for each index in batch and append to each batch array
        for indx in batch_indices[1:]:
            # get window of input parameters for index according to training_days
            windowed_data = parameters_full[indx - training_days:indx, :, :].unsqueeze(0)
            # add to batch array of input parameters
            batch_data_windows = torch.cat((batch_data_windows, windowed_data), dim=0)
            # get corresponding label day for index according to prediction_day
            windowed_label = labels_full[indx + prediction_day, :, :].unsqueeze(0)
            # add to batch array of labels
            batch_label_windows = torch.cat((batch_label_windows, windowed_label), dim=0)
        # generate array of corresponding masks for each index (if applicable)
        if include_masks:
            for indx in batch_indices[1:]:
                windowed_mask = masks_full[indx + prediction_day, :, :].unsqueeze(0)
                batch_mask_windows = torch.cat((batch_mask_windows, windowed_mask), dim=0)

    # return as list so code doesn't break if masks are not requested
    return [batch_data_windows, batch_label_windows, batch_mask_windows]


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
