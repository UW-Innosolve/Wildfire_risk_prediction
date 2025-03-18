from model_evaluation.model_lossfunctions import binary_cross_entropy_loss as bce_loss
from model_classes.lstm import LSTM_3D
from data_preprocessing.windowing import batched_indexed_windows, reshape_data

import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import torch
import logging

import tensorboard as tb


# Configure logging: INFO level logs progress, DEBUG could be used for more details.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


features = ['10u', '10v', '2d', '2t', 'cl', 'cvh',
            'cvl', 'fal', 'lai_hv', 'lai_lv', 'lsm', 'slt', 'sp', 'src', 'stl1',
            'stl2', 'stl3', 'stl4', 'swvl1', 'swvl2', 'swvl3', 'swvl4', 'tvh',
            'tvl', 'z', 'e', 'pev', 'slhf', 'sshf', 'ssr', 'ssrd', 'str', 'strd',
            'tp', 'is_fire_day', 'lightning_count', 'absv_strength_sum',
            'multiplicity_sum', 'railway_count', 'power_line_count',
            'highway_count', 'aeroway_count', 'waterway_count']
target_column = 'is_fire_day'

rawdata_path = "/Users/teodoravujovic/Desktop/data/firebird/march13_pull/fb_raw_data_201407.csv"

# load raw data into pandas
rawdata_df = pd.read_csv(rawdata_path)

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

reshaped_data, reshaped_labels = reshape_data(rawdata_df, features, target_column)


# TODO create a training_parameters json or something similar to make tracking easier?
def main(dataset=reshaped_data, labels=reshaped_labels, training_parameters={"batch_size": 4,"num_epochs": 5,"learning_rate": 0.003,"features": 42, "num_training_days": 5, "prediction_day":5}):
    #
    batch_size = training_parameters['batch_size']
    num_epochs = training_parameters['num_epochs']
    learning_rate = training_parameters['learning_rate']
    features = training_parameters['features']
    num_training_days = training_parameters['num_training_days']
    prediction_day = training_parameters['prediction_day']

    # Define the list of features based on our dataset headers.
    logging.info(f"Selected features: {features}")
    logging.info(f"Target variable: {target_column}")

    data = torch.Tensor(reshaped_data)
    labels = torch.Tensor(reshaped_labels)

    test_indices_list = []
    for i in range(6, 25):
        test_indices_list.append(i)
    test_indices_np = np.asarray(test_indices_list)

    # Split the data; apply SMOTE for balancing minority class (fire days).
    logging.info("Splitting data into training and test sets using day index...")
    # TODO complete train test splitting
    X_train, X_test, y_train, y_test = train_test_split(test_indices_np, test_indices_np, train_size=0.85)

    # create model
    model = LSTM_3D(input_channels=features, hidden_size=64, dropout_rate=0.02)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        np.random.shuffle(X_train)
        np.random.shuffle(X_test)
        print(X_train)
        print(X_test)
        batches = X_train.reshape(int(len(X_train)/batch_size), batch_size)

        for batch in batches:
            print(f'Epoch {epoch}, Batch {batch}')
            optimizer.zero_grad()
            inputs, targets = batched_indexed_windows(batch, data, labels, num_training_days, prediction_day)
            outputs = model(inputs)  # forward pass
            # outputs = outputs.squeeze(2)  # remove extra dimension
            loss = bce_loss(outputs, targets)
            loss.backward()

        optimizer.step()
        # if (epoch) % 10 == 0:
        #     print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


if __name__ == "__main__":
    main()
