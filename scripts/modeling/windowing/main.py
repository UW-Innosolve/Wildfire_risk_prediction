### OVERALL PROCESS
## pull data and labels
## determine number of lines of latitude and longitude
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


# load data (path local to Teo's machine for now)


# get train and labels dataframes
train_table = load_data(train_path)
labels_df = pd.read_csv(train_labels_path)


def load_sample(table, series_id):
    expr = pc.field('series_id') == series_id
    sample = table.filter(expr)
    df = sample.to_pandas()
    return df


# process labels_df to fill all nan values with None
labels_df = labels_df.fillna(value='None')

# create a label series for one series_id
# periods of awake will be labelled 0, periods of asleep will be labelled 1.
def create_labels(events, steps, label_length):
    label = [None] * label_length

    # set values of labels prior to first event
    if not (steps[0] == 'None'):
        label[:int(steps[0])] = [1 - events[0]] * int(steps[0])

    # set values of labels from events 1:n-1
    for i in range(1, len(events)):
        print(i)
        evnt = events[i - 1]
        prv_step = steps[i - 1]
        cur_step = steps[i]
        if not (cur_step == 'None'):
            if not (prv_step == 'None'):
                prv_step = int(prv_step)
                cur_step = int(cur_step)
                print(prv_step, cur_step)
                label[prv_step:cur_step] = [evnt] * (cur_step - prv_step)

    # set values of labels after last event
    if not (steps[-1] == 'None'):
        label[int(steps[-1]):] = [events[-1]] * (label_length - int(steps[-1]))

    return label

# get all unique series_id
list_ids = train_table['series_id'].unique()

# get all complete series (no None values in step column of label df)
useful_series = []
for series_id in list_ids:
    sample_label_df = labels_df[labels_df['series_id'] == str(series_id)]
    step = sample_label_df['step'].to_numpy()
    if np.count_nonzero(step == 'None') == 0:
        print(series_id)
        useful_series.append(series_id)

# create empty array for full series of data
train = []

# pull full series data for each complete sample
# print statements included for debugging purposes
for series_id in useful_series:
    sample_df = load_sample(train_table, series_id)
    anglez = sample_df['anglez'].to_numpy()
    enmo = sample_df['enmo'].to_numpy()
    sample_label_df = labels_df[labels_df['series_id'] == str(series_id)]
    event = sample_label_df['event'].to_numpy()
    event = np.asarray([1 if x == 'onset' else 0 for x in event])
    step = sample_label_df['step'].to_numpy()
    label = create_labels(event, step, len(anglez))
    sample = [anglez, enmo, label]
    train.append(sample)


# works as expected
series_in_chunks = []
series_in_chunks_labels = []

# if there are none values, move ahead 300 steps (25 minutes) until at a time until there are no None values
# series selected are complete, so we don't need to worry about the above comment for now
for i in range(len(useful_series)):
    print(i)
    series_pieces = []
    sample = np.asarray(train[i])
    length = len(sample[0])
    start_step = 0
    end_step = 8192
    while end_step < length:
        count_null_labels = np.count_nonzero(sample[2][start_step:end_step] == None)
        if count_null_labels == 0:
            series_in_chunks.append(np.asarray([sample[0][start_step:end_step], sample[1][start_step:end_step]]))
            series_in_chunks_labels.append(np.asarray(sample[2][start_step:end_step]))
        else:
            print('FLAG: Nulls found')
        start_step += 300
        end_step += 300

series_in_chunks = np.asarray(series_in_chunks)
series_in_chunks_labels = np.asarray(series_in_chunks_labels)