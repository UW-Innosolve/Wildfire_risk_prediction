from model_evaluation.model_lossfunctions import binary_cross_entropy_loss as bce_loss
from model_classes.lstm import LSTM_3D
from data_preprocessing.windowing import batched_indexed_windows, reshape_data
# from data_preprocessing.csv_aggreg import csv_aggregate
# from model_evaluation.nn_model_metrics import evaluate


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


def get_indices(data_df, train_range, test_range, start_day='02-24', end_day='09-25'):
    '''
    Because we use the day we predict from (not the day we are predicting) as the index, we must subtract 5 days from March 1 and 5 days from Sept 31
    '''
    train_indices = []
    test_indices = []

    dates = data_df['date'].unique().tolist()

    for year in range(train_range[0], train_range[1] + 1):
        start_index = dates.index(f"{year}-{start_day}")
        end_index = dates.index(f"{year}-{end_day}")
        for i in range(start_index, end_index + 1):
            train_indices.append(i)

    if type(test_range) == int:
        start_index = dates.index(f"{test_range}-{start_day}")
        end_index = dates.index(f"{test_range}-{end_day}")
        for i in range(start_index, end_index + 1):  # +1 is to catch all days (and not cut off the last one)
            test_indices.append(i)
    else:
        for year in range(test_range[0], test_range[1] + 1): # +1 to not cut off the last year
            start_index = dates.index(f"{year}-{start_day}")
            end_index = dates.index(f"{year}-{end_day}")
            for i in range(start_index, end_index + 1):  # +1 is to catch all days (and not cut off the last one)
                test_indices.append(i)

    return np.asarray(train_indices), np.asarray(test_indices)


# TODO create a training_parameters json or something similar to make tracking easier
# TODO update parameters to pull from a json file
# TODO update to run on a device (i.e. cpu or gpu)
def main(training_parameters={"batch_size": 10,
                              "num_epochs": 10,
                              "learning_rate": 0.05,
                              "num_training_days": 14,
                              "prediction_day":5,
                              "hidden_size": 64,
                              "experiment_name":"testrun",
                              "test_range": (2024),
                              "train_range": (2006, 2023)},
         rawdata_path='/home/tvujovic/scratch/firebird/processed_data.csv',
         # rawdata_path='/Users/teodoravujovic/Desktop/code/firebird/processed_data.csv',
         device_set='cuda'):
    # load training parameters
    batch_size = training_parameters['batch_size']
    num_epochs = training_parameters['num_epochs']
    learning_rate = training_parameters['learning_rate']
    # num_features = training_parameters['features'] # now obtained from dataframe
    num_training_days = training_parameters['num_training_days']
    prediction_day = training_parameters['prediction_day']
    hidden_size = training_parameters['hidden_size']
    experiment_name = training_parameters['experiment_name']
    checkpoint_dir = f'./checkpoints/{experiment_name}/'
    train_range = training_parameters['train_range']
    test_range = training_parameters['test_range']
    logging.info(f"Training parameters set successfully")

    # set device
    if device_set == 'cuda':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logging.info(f"Device set to cuda, running on GPU")
        else:
            device = torch.device('cpu')
            logging.info(f"No GPU available! Device set to CPU")
    else:
        device = torch.device('cpu')
        logging.info(f"Device set to CPU")

    # load data from df
    rawdata_df = pd.read_csv(rawdata_path) #.to(device)
    logging.info(f"Dataset csv file loaded into dataframe successfully")
    # assert rawdata_df.isna().sum() == 0 # assert no nulls in dataframe
    features = rawdata_df.columns[3:].array
    target_column = 'is_fire_day'
    num_features = len(features)
    logging.info(f"Selected features: {features}")
    logging.info(f"Target variable: {target_column}")

    # get train and test set indices
    train_indices, test_indices = get_indices(rawdata_df, train_range, test_range) # set for fire season only unless changed
    logging.info(f"Indexing completed, train_indices and test_indices sets created successfully")

    # reshape data into 2-D
    # TODO update reshaping so that its done in torch
    # TODO update reshaping to be done using dates and not indices
    reshaped_data, reshaped_labels = reshape_data(rawdata_df, features, target_column, device_set)
    # logging
    logging.info(f"Successfully reshaped all features")

    # remove after converting reshape function to torch
    data = torch.Tensor(reshaped_data)
    labels = torch.Tensor(reshaped_labels)

    # set tensorboard writer directory
    writer = SummaryWriter(log_dir=checkpoint_dir)
    # logging
    logging.info(f"Tensorboard output directory configured to {checkpoint_dir}")

    # Split the data; apply SMOTE for balancing minority class (fire days).
    logging.info("Splitting data into training, validation, and test sets using day index")
    # TODO complete train test splitting
    # split train_indices into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(train_indices, train_indices, train_size=0.85)
    # set test_indices as test set
    X_test, y_test = test_indices, test_indices
    logging.info(f"Data split successfully, train_set size - {len(X_train)}, val_set size - {len(X_val)}, test_set size - {len(X_test)}")

    # set size of validation batch
    if device_set == 'cuda' and torch.cuda.is_available():
        val_batch_size = len(X_val) # use full validation set each time if GPU used
    else:
        val_batch_size = 20 # use small batch of validation set each time if CPU used

    # create model
    model = LSTM_3D(input_channels=num_features, hidden_size=hidden_size, dropout_rate=0.02)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    batch_num = 0
    logging.info("Model created successfully")
    samples_per_epoch = len(X_train) - (len(X_train) % batch_size)

    for epoch in range(num_epochs):
        np.random.shuffle(X_train)
        batches = X_train[:samples_per_epoch].reshape(int(len(X_train)/batch_size), batch_size)

        for batch in batches:
            print(f'Epoch {epoch}, Batch Number {batch_num}, Batch Indices {batch}')
            optimizer.zero_grad()
            inputs, targets = batched_indexed_windows(batch, data, labels, num_training_days, prediction_day)
            outputs = model(inputs)
            loss = bce_loss(outputs, targets)
            if (batch_num % 20) == 0:
                with torch.no_grad():
                    np.random.shuffle(X_val)
                    label_batch = X_val[:val_batch_size] # TODO: allow to test the entire validation set at once in gpu implementation
                    test_inputs, test_targets = batched_indexed_windows(label_batch, data, labels, num_training_days, prediction_day)
                    test_predictions = model(test_inputs)
                    # test_metrics = evaluate(test_predictions, test_targets)
                    test_loss = bce_loss(test_predictions, test_targets)
                    print(f"Validation Batch Loss: Batch Num {batch_num}, Loss: {test_loss}")
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

    # Option 1: Save the model's state_dict (recommended)
        torch.save(model.state_dict(), f'{checkpoint_dir}/model_epoch_{epoch}.pth')
        logging.info(f"Model state dictionary for epoch {epoch} saved to: {checkpoint_dir}/model_epoch_{epoch}.pth")

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
        torch.save(checkpoint, f'{checkpoint_dir}/checkpoint_epoch_{epoch}.pth')
        logging.info(f"Model checkpoint for epoch {epoch} saved to: {checkpoint_dir}/checkpoint_epoch_{epoch}.pth")

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
    # loaded_model.load_state_dict(loaded_checkpoint['model_state_dict'])
    # optimizer = torch.optim.Adam(loaded_model.parameters())  # or your optimizer
    # optimizer.load_state_dict(loaded_checkpoint['optimizer_state_dict'])
    # # Access other saved information:
    # # epoch = loaded_checkpoint['epoch']
    # # loss = loaded_checkpoint['loss']
    #
    # loaded_model.eval()  # or loaded_model.train() depending on your usecase.

if __name__ == "__main__":
    main()
