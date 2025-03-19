from model_evaluation.model_lossfunctions import binary_cross_entropy_loss as bce_loss
from model_classes.lstm import LSTM_3D
from data_preprocessing.windowing import batched_indexed_windows, reshape_data
from data_preprocessing.csv_aggreg import csv_aggregate
from model_evaluation.nn_model_metrics import evaluate
import os


import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import torch
import logging
from typing import Dict

from torch.utils.tensorboard.writer import SummaryWriter


def tb_optimizer(
        writer: SummaryWriter,
        losses_dict: Dict[str, torch.Tensor],
        step: int,
) -> None:
    for loss_name, loss in losses_dict.items():
        writer.add_scalar(loss_name, loss, global_step=step)


# Configure logging: INFO level logs progress, DEBUG could be used for more details.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')


features = ['10u', '10v', '2d', '2t', 'cl', 'cvh',
            'cvl', 'fal', 'lai_hv', 'lai_lv', 'lsm', 'slt', 'sp', 'src', 'stl1',
            'stl2', 'stl3', 'stl4', 'swvl1', 'swvl2', 'swvl3', 'swvl4', 'z',
            'e', 'pev', 'slhf', 'sshf', 'ssr', 'ssrd', 'str', 'strd',
            'tp', 'is_fire_day',
            'railway_count', 'power_line_count',
            'highway_count', 'aeroway_count', 'waterway_count']
# REMOVED 'multiplicity_sum' 'tvh', 'tvl', 'lightning_count', 'absv_strength_sum',

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


# rawdata_path = "/Users/teodoravujovic/Desktop/data/firebird/march13_pull/fb_raw_data_201407.csv"
# rawdata_path = "/Users/teodoravujovic/Downloads/fb_raw_data_2006-2024_split/fb_raw_data_5.csv"
# rawdata_path = '/Users/teodoravujovic/Desktop/code/firebird/lstm_training/Wildfire_risk_prediction/scripts/data_processing/processed_data_no_cffdrs_5.csv'
# data_dir = '/Users/teodoravujovic/Desktop/code/firebird/lstm_training/Wildfire_risk_prediction/scripts/modeling/raw_data'
data_dir = '/Users/teodoravujovic/Desktop/data/firebird/fb_complete_raw_output_20250316'
# rawdata_df = csv_aggregate(data_dir)
big_csv_path = os.path.join(data_dir, 'big.csv')
if os.path.exists(big_csv_path):
    rawdata_df = pd.read_csv(big_csv_path)
else:
    rawdata_df = csv_aggregate(data_dir)
    rawdata_df.to_csv(big_csv_path, index=False)

# load raw data into pandas
# rawdata_df = pd.read_csv(rawdata_path)
# features = rawdata_df.columns[3:].tolist()
target_column = 'is_fire_day'

reshaped_data, reshaped_labels = reshape_data(rawdata_df, features, target_column)


usable_ranges = [range(54, 273), range(419, 638), range(784, 1003), range(1149, 1368), range(1514, 1733), range(1879, 2098), range(2244, 2463), range(2609, 2828), range(2974, 3193), range(3339, 3558), range(3704, 3923), range(4069, 4288), range(4313, 4532), range(4678, 4897), range(5043, 5262), range(5408, 5623)] #, range(5773, 5989)] #, range(6138, 6355)]#, range(6503, 6722)]
usable_indices = []
test_indices = []
for usable_range in usable_ranges:
    for i in usable_range:
            usable_indices.append(i)
for i in range(5773, 5989): # 2022 fire season
    test_indices.append(i)
fireseason_indices_np = np.asarray(usable_indices)
testfireseason_indices_np = np.asarray(test_indices)


# TODO create a training_parameters json or something similar to make tracking easier?
def main(dataset=reshaped_data, labels=reshaped_labels, training_parameters={"batch_size": 10,"num_epochs": 8,"learning_rate": 0.005,"features": len(features), "num_training_days": 14, "prediction_day":5, "hidden_size": 64, "experiment_name":"rawtrain_8"}):
    batch_size = training_parameters['batch_size']
    num_epochs = training_parameters['num_epochs']
    learning_rate = training_parameters['learning_rate']
    features = training_parameters['features']
    num_training_days = training_parameters['num_training_days']
    prediction_day = training_parameters['prediction_day']
    hidden_size = training_parameters['hidden_size']
    experiment_name = training_parameters['experiment_name']
    checkpoint_dir = f'./checkpoints/{experiment_name}/'

    # Define the list of features based on our dataset headers.
    logging.info(f"Selected features: {features}")
    logging.info(f"Target variable: {target_column}")

    data = torch.Tensor(reshaped_data)
    labels = torch.Tensor(reshaped_labels)
    writer = SummaryWriter(log_dir=checkpoint_dir)

    # test_indices_list = []
    # for i in range(20, 550):
    #     test_indices_list.append(i)
    # test_indices_np = np.asarray(test_indices_list)

    # Split the data; apply SMOTE for balancing minority class (fire days).
    logging.info("Splitting data into training and test sets using day index...")
    # TODO complete train test splitting
    # X_train, X_test, y_train, y_test = train_test_split(fireseason_indices_np, testfireseason_indices_np, train_size=0.8)
    # X_train, X_test, y_train, y_test = train_test_split(test_indices_np, test_indices_np, train_size=0.85)
    X_train, X_test, y_train, y_test = fireseason_indices_np, testfireseason_indices_np, fireseason_indices_np, testfireseason_indices_np

    # create model
    loaded_checkpoint = torch.load('checkpoint.pth')
    model = LSTM_3D(input_channels=features, hidden_size=64, dropout_rate=0.02)
    model.load_state_dict(loaded_checkpoint['model_state_dict'])
    optimizer = torch.optim.Adam(model.parameters())  # or your optimizer
    batch_num = 0
    full_outputs = []

    for epoch in range(num_epochs):
        np.random.shuffle(X_train)
        # print(X_train)
        # print(X_test)
        batches = X_train.reshape(int(len(X_train)/batch_size), batch_size)

        for batch in batches:
            print(f'Epoch {epoch}, Batch {batch}')
            optimizer.zero_grad()
            inputs, targets = batched_indexed_windows(batch, data, labels, num_training_days, prediction_day)
            outputs = model(inputs)  # forward pass]
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1, norm_type=2, error_if_nonfinite=True)
            loss = bce_loss(outputs, targets)
            if (batch_num) == 0:
                with torch.no_grad():
                    # np.random.shuffle(X_test)
                    for i in X_test:
                        label_batch = [i]
                        test_inputs, test_targets = batched_indexed_windows(label_batch, data, labels, num_training_days, prediction_day)
                        test_predictions = model(test_inputs, eval=True)
                        # test_metrics = evaluate(test_predictions, test_targets)
                        full_outputs.append(test_predictions)
                        test_loss = bce_loss(test_predictions, test_targets)
                        print(f'Test loss: {test_loss}')
                        print(f"Validation Batch Loss: Batch Num {batch_num}, Loss: {test_loss}")
                        metrics_dict = {"training_bce_loss": loss.item(),
                                        "validation_bce_loss": test_loss.item()}#,
                                        # "accuracy": test_metrics["accuracy"],
                                        # "precision": test_metrics["precision"],
                                        # "recall": test_metrics["recall"],
                                        # "f1_score": test_metrics["f1_score"]}
                                        # # "roc_auc": test_metrics["roc_auc"]}
                        tb_optimizer(writer=writer, losses_dict=metrics_dict, step=batch_num)
            return full_outputs
            loss.backward()
            batch_num += 1
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1, norm_type=2, error_if_nonfinite=True)

            optimizer.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    # import torch

    # Assuming 'model' is your PyTorch model
    # and 'optimizer' is your optimizer (optional)

    # Option 1: Save the model's state_dict (recommended)
    torch.save(model.state_dict(), 'model.pth')

    # Option 2: Save the entire model (not recommended for production)
    # This saves the model's architecture and weights.
    # It can be problematic if the model class changes.
    # torch.save(model, 'model_full.pth')

    # Option 3: Save a checkpoint containing model and optimizer state.
    # This is useful for resuming training.
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),  # If you have an optimizer
        # Add other relevant information like epoch, loss, etc.
    }
    torch.save(checkpoint, 'checkpoint.pth')

    # # Example of loading the model's state_dict:
    # loaded_model = YourModelClass(*args, **kwargs)  # Instantiate your model
    # loaded_model.load_state_dict(torch.load('model.pth'))
    # loaded_model.eval()  # Important: set to evaluation mode if you're doing inference
    #
    # # Example of loading a full model (not recommended)
    # # loaded_full_model = torch.load('model_full.pth')
    # # loaded_full_model.eval()
    #
    # # Example of loading a checkpoint:
    # loaded_checkpoint = torch.load('checkpoint.pth')
    # loaded_model = YourModelClass(*args, **kwargs)  # Instantiate your model
    # loaded_model.load_state\
    #
    # _dict(loaded_checkpoint['model_state_dict'])
    # optimizer = torch.optim.Adam(loaded_model.parameters())  # or your optimizer
    # optimizer.load_state_dict(loaded_checkpoint['optimizer_state_dict'])
    # # Access other saved information:
    # # epoch = loaded_checkpoint['epoch']
    # # loss = loaded_checkpoint['loss']
    #
    # loaded_model.eval()  # or loaded_model.train() depending on your usecase.


full_outputs = main()
print('all_good')


list_of_dates = pd.date_range(start="2022-02-25", end="2022-09-26", freq="D")
lat_range = [49, 60]
long_range = [-120, -110]
grid_resolution = 0.30
output_df = vis_csv_maker(list_of_dates, list_of_arrays, lat_range, long_range, grid_resolution)

if __name__ == "__main__":
    main()
