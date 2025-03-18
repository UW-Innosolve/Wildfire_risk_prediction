from model_evaluation.model_lossfunctions import binary_cross_entropy_loss as bce_loss
from model_classes.lstm import LSTM_3D
from data_preprocessing.windowing import batched_indexed_windows, reshape_data
from model_evaluation.nn_model_metrics import evaluate

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
    checkpoint_dir = './checkpoints/'

    # Define the list of features based on our dataset headers.
    logging.info(f"Selected features: {features}")
    logging.info(f"Target variable: {target_column}")

    data = torch.Tensor(reshaped_data)
    labels = torch.Tensor(reshaped_labels)
    writer = SummaryWriter(log_dir=checkpoint_dir)

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
        # print(X_train)
        # print(X_test)
        batches = X_train.reshape(int(len(X_train)/batch_size), batch_size)
        batch_num = 0

        for batch in batches:
            print(f'Epoch {epoch}, Batch {batch}')
            optimizer.zero_grad()
            inputs, targets = batched_indexed_windows(batch, data, labels, num_training_days, prediction_day)
            outputs = model(inputs)  # forward pass
            loss = bce_loss(outputs, targets)
            if (batch_num % 10) == 0:
                with torch.no_grad():
                    np.random.shuffle(X_test)
                    label_batch = X_test[:4]
                    test_inputs, test_targets = batched_indexed_windows(label_batch, data, labels, num_training_days, prediction_day)
                    test_predictions = model(test_inputs)
                    # test_metrics = evaluate(test_predictions, test_targets)
                    test_loss = bce_loss(test_predictions, test_targets)
                    metrics_dict = {"training_bce_loss": loss.item(),
                                    "validation_bce_loss": test_loss.item()}#,
                                    # "accuracy": test_metrics["accuracy"],
                                    # "precision": test_metrics["precision"],
                                    # "recall": test_metrics["recall"],
                                    # "f1_score": test_metrics["f1_score"]}
                                    # # "roc_auc": test_metrics["roc_auc"]}
                    tb_optimizer(writer=writer, losses_dict=metrics_dict, step=batch_num)
            loss.backward()
            batch_num += 1

            optimizer.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


if __name__ == "__main__":
    main()
