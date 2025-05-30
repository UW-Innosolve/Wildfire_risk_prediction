import numpy as np


def get_indices(data_df, train_range, test_range, start_day='02-24', end_day='09-25'):
    """
    Description: Returns an array of all possible sample indices for both the training and test sets according to specified ranges.
    All possible dates are assigned an index starting from 1 through the number of unique dates in the dataframe.
    These are the indices assigned to each sample and utilized for dataloading during training and evaluation.

    Parameters:
        data_df: pd.DataFrame
        train_range: tuple
            Specified years to use for training and validation
        test_range: tuple or int
            Specified years to use for testing
        start_day: str
            Default: February 24
            First day of each year to consider as 'fire season'
        end_day: str
            Default: September 25
            Last day of each year to consider as 'fire season'

    Returns:
        np.ndarray, np.ndarray
            Array of indices of all potential training and test samples according to specified ranges.

    Notes:
        - Because we use the day we predict from (not the day we are predicting) as the index, we must subtract 5 days from March 1 and 5 days from September 31
        - February 29 is ignored and skipped due to issues with Pandas. Will be corrected in the future. All years are treated as having 365 days.
    """
    # instantiate index lists
    train_indices = []
    test_indices = []

    # get list of all unique dates in dataframe
    dates = data_df['date'].unique().tolist()

    # get indices of all days between start_day and end_day for each year in train_range
    for year in range(train_range[0], train_range[1] + 1):
        start_index = dates.index(f"{year}-{start_day}")
        end_index = dates.index(f"{year}-{end_day}")
        for i in range(start_index, end_index + 1):
            train_indices.append(i) # append indices to training set

    # get indices of all days between start_day and end_day for each year in test_range
    #   test range may be a single year
    if type(test_range) == int:
        start_index = dates.index(f"{test_range}-{start_day}")
        end_index = dates.index(f"{test_range}-{end_day}")
        for i in range(start_index, end_index + 1):  # +1 is to catch all days (and not cut off the last one)
            test_indices.append(i) # append indices to test set
    else:
        for year in range(test_range[0], test_range[1] + 1): # +1 to not cut off the last year
            start_index = dates.index(f"{year}-{start_day}")
            end_index = dates.index(f"{year}-{end_day}")
            for i in range(start_index, end_index + 1):  # +1 is to catch all days (and not cut off the last one)
                test_indices.append(i) # append indices to test set

    # convert lists to arrays and return
    return np.asarray(train_indices), np.asarray(test_indices)