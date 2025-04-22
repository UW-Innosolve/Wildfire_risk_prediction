from model_evaluation.model_lossfunctions import binary_cross_entropy_loss as bce_loss
from model_classes.lstm import LSTM_3D
from data_preprocessing.windowing import batched_indexed_windows, reshape_data
from model_evaluation.nn_model_metrics import evaluate, calculate_metrics, create_empty_metrics_dict
from visualize import set_colour_scheme, plot_target_vs_predictions
from common import get_indices

import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import torch
import logging
from typing import Dict
import os
import json
import typer
import matplotlib.pyplot as plt
import matplotlib as mpl

from torch.utils.tensorboard.writer import SummaryWriter


# set default colourmap and boundary norms
cmap_default, norm_default = set_colour_scheme()


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

app = typer.Typer()


@app.command()
def main(parameter_set_key:str='default',
         training_parameter_json='./training_params.json',
         rawdata_path='/home/tvujovic/scratch/firebird/processed_data.csv',
         # rawdata_path='/Users/teodoravujovic/Desktop/code/firebird/processed_data.csv',
         device_set='cuda',
         include_masks=True, # TODO make sure this is included everywhere for modularity, make sure it doesnt slow things down too much
         mask_size=2,
         default_threshold_value=0.515,
         log_thresholding=True,
         generate_predictions=True,
         from_checkpoint=True,
         checkpoint_directory='/Users/teodoravujovic/Desktop/data/firebird/thresholding_experiments/',):

    # open training_parameters json file
    with open(training_parameter_json) as json_data:
        training_parameters = json.load(json_data)[parameter_set_key]
        json_data.close()

    # load training parameters
    batch_size = training_parameters['batch_size']
    num_epochs = training_parameters['num_epochs']
    learning_rate = training_parameters['learning_rate']
    num_training_days = training_parameters['num_training_days']
    prediction_day = training_parameters['prediction_day']
    hidden_size = training_parameters['hidden_size']
    experiment_name = training_parameters['experiment_name']
    checkpoint_dir = f'./checkpoints/{experiment_name}/'
    train_range = (training_parameters['train_range_start'], training_parameters['train_range_end'])
    test_range = training_parameters['test_range']
    logging.info(f"Training parameters set successfully")
    logging.info(f"Training Sample Length - {num_training_days}")
    logging.info(f"Prediction day - {prediction_day}")

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
    rawdata_df = pd.read_csv(rawdata_path) #[:1377510] #.to(device)
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
    # TODO update reshaping to be done using dates and not indices
    reshaped_data, reshaped_labels, reshaped_masks = reshape_data(rawdata_df, features, target_column, device_set, include_masks=include_masks, tolerance=mask_size)
    logging.info(f"Successfully reshaped all features")

    # remove after converting reshape function to torch
    data = torch.Tensor(reshaped_data).to(device)
    labels = torch.Tensor(reshaped_labels).to(device)
    masks = torch.Tensor(reshaped_masks).to(device)

    # TODO: make it modular which subdirectories to create
    # set tensorboard writer directory and subdirectories
    writer = SummaryWriter(log_dir=f"{checkpoint_dir}threshold_0515")
    writer_fire = SummaryWriter(log_dir=f"{checkpoint_dir}fire_threshold_0515")
    writer_region = SummaryWriter(log_dir=f"{checkpoint_dir}region_threshold_0515")
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
    # if device_set == 'cuda' and torch.cuda.is_available():
    #     val_batch_size = len(X_val) # use full validation set each time if GPU used
    # else:
    #     val_batch_size = 20 # use small batch of validation set each time if CPU used
    val_batch_size = 30
    val_batch_nums = len(X_val) // val_batch_size

    # create model
    model = LSTM_3D(input_channels=num_features, hidden_size=hidden_size, dropout_rate=0.02).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    batch_num = 0
    logging.info("Model created successfully")
    samples_per_epoch = len(X_train) - (len(X_train) % batch_size)

    # TODO fix so this is modular
    batch_flat_shape = 1258 * batch_size
    batch_flat_shape_val = 1258 * val_batch_size

    # TODO: fix so its not hardcoded
    # create tensorboard writers for all other test threshold values
    if log_thresholding:
        writer_010 = SummaryWriter(log_dir=f"{checkpoint_dir}threshold_010")
        writer_015 = SummaryWriter(log_dir=f"{checkpoint_dir}threshold_015")
        writer_020 = SummaryWriter(log_dir=f"{checkpoint_dir}threshold_020")
        writer_030 = SummaryWriter(log_dir=f"{checkpoint_dir}threshold_030")
        writer_035 = SummaryWriter(log_dir=f"{checkpoint_dir}threshold_035")
        writer_040 = SummaryWriter(log_dir=f"{checkpoint_dir}threshold_040")
        writer_045 = SummaryWriter(log_dir=f"{checkpoint_dir}threshold_045")
        writer_050 = SummaryWriter(log_dir=f"{checkpoint_dir}threshold_050")
        writer_051 = SummaryWriter(log_dir=f"{checkpoint_dir}threshold_051")
        writer_053 = SummaryWriter(log_dir=f"{checkpoint_dir}threshold_053")
        writer_055 = SummaryWriter(log_dir=f"{checkpoint_dir}threshold_055")
        writer_060 = SummaryWriter(log_dir=f"{checkpoint_dir}threshold_060")
        writer_070 = SummaryWriter(log_dir=f"{checkpoint_dir}threshold_070")

        writer_fire_010 = SummaryWriter(log_dir=f"{checkpoint_dir}fire_threshold_010")
        writer_fire_015 = SummaryWriter(log_dir=f"{checkpoint_dir}fire_threshold_015")
        writer_fire_020 = SummaryWriter(log_dir=f"{checkpoint_dir}fire_threshold_020")
        writer_fire_030 = SummaryWriter(log_dir=f"{checkpoint_dir}fire_threshold_030")
        writer_fire_035 = SummaryWriter(log_dir=f"{checkpoint_dir}fire_threshold_035")
        writer_fire_040 = SummaryWriter(log_dir=f"{checkpoint_dir}fire_threshold_040")
        writer_fire_045 = SummaryWriter(log_dir=f"{checkpoint_dir}fire_threshold_045")
        writer_fire_050 = SummaryWriter(log_dir=f"{checkpoint_dir}fire_threshold_050")
        writer_fire_051 = SummaryWriter(log_dir=f"{checkpoint_dir}fire_threshold_051")
        writer_fire_053 = SummaryWriter(log_dir=f"{checkpoint_dir}fire_threshold_053")
        writer_fire_055 = SummaryWriter(log_dir=f"{checkpoint_dir}fire_threshold_055")
        writer_fire_060 = SummaryWriter(log_dir=f"{checkpoint_dir}fire_threshold_060")
        writer_fire_070 = SummaryWriter(log_dir=f"{checkpoint_dir}fire_threshold_070")

        writer_region_010 = SummaryWriter(log_dir=f"{checkpoint_dir}region_threshold_010")
        writer_region_015 = SummaryWriter(log_dir=f"{checkpoint_dir}region_threshold_015")
        writer_region_020 = SummaryWriter(log_dir=f"{checkpoint_dir}region_threshold_020")
        writer_region_030 = SummaryWriter(log_dir=f"{checkpoint_dir}region_threshold_030")
        writer_region_035 = SummaryWriter(log_dir=f"{checkpoint_dir}region_threshold_035")
        writer_region_040 = SummaryWriter(log_dir=f"{checkpoint_dir}region_threshold_040")
        writer_region_045 = SummaryWriter(log_dir=f"{checkpoint_dir}region_threshold_045")
        writer_region_050 = SummaryWriter(log_dir=f"{checkpoint_dir}region_threshold_050")
        writer_region_051 = SummaryWriter(log_dir=f"{checkpoint_dir}region_threshold_051")
        writer_region_053 = SummaryWriter(log_dir=f"{checkpoint_dir}region_threshold_053")
        writer_region_055 = SummaryWriter(log_dir=f"{checkpoint_dir}region_threshold_055")
        writer_region_060 = SummaryWriter(log_dir=f"{checkpoint_dir}region_threshold_060")
        writer_region_070 = SummaryWriter(log_dir=f"{checkpoint_dir}region_threshold_070")

        writer_list = [writer_045, writer_050, writer_051, writer_053, writer_055, writer_060, writer_070]

    for epoch in range(num_epochs):
        np.random.shuffle(X_train)
        batches = X_train[:samples_per_epoch].reshape(int(len(X_train)/batch_size), batch_size)

        for batch in batches:
            print(f'Epoch {epoch}, Batch Number {batch_num}, Batch Indices {batch}')
            optimizer.zero_grad()
            # get batched windows for inputs, targets, and fire region masks
            # TODO determine how much slower it is to assign inputs targets and masks like this
            batch_windows = batched_indexed_windows(batch, data, labels, num_training_days, prediction_day, include_masks=True, masks_full=masks)
            inputs, targets, regions = batch_windows[0], batch_windows[1], batch_windows[2]
            outputs = model(inputs)

            # calculate losses
            full_loss = bce_loss(outputs, targets)
            # TODO figure out why fire_loss is getting so big
            fire_loss = bce_loss(outputs*targets, targets)
            region_loss = bce_loss(outputs*regions, targets)
            loss = (full_loss * 0.2) + (region_loss * 0.8)
            print(f"Scaled Loss: {loss}, Fire_Region Loss: {region_loss}, Fire Loss: {fire_loss}, Full Loss: {full_loss}")

            if (batch_num % 20) == 0:
                with torch.no_grad():
                    # calculate training metrics for most recent batch
                    train_metrics_dict = evaluate(outputs, targets, batch_flat_shape, threshold_value=default_threshold_value)
                    print(f"Accuracy: {train_metrics_dict['accuracy']}, Precision: {train_metrics_dict['precision']}, Recall: {train_metrics_dict['recall']}, F1: {train_metrics_dict['f1']}")

                    # set total validation losses and metrics to 0, we will average over the entire validation set later
                    val_scaled_loss = 0
                    val_full_loss = 0
                    val_fire_loss = 0
                    val_fire_region_loss = 0

                    # random shuffle the order of the validation set
                    # TODO: does this matter since we aren't learning anyways?
                    np.random.shuffle(X_val)

                    metrics = create_empty_metrics_dict()
                    metrics_fire = create_empty_metrics_dict()
                    metrics_regions = create_empty_metrics_dict()

                    if log_thresholding:
                        metrics_010 = create_empty_metrics_dict()
                        metrics_fire_010 = create_empty_metrics_dict()
                        metrics_regions_010 = create_empty_metrics_dict()

                        metrics_015 = create_empty_metrics_dict()
                        metrics_fire_015 = create_empty_metrics_dict()
                        metrics_regions_015 = create_empty_metrics_dict()

                        metrics_020 = create_empty_metrics_dict()
                        metrics_fire_020 = create_empty_metrics_dict()
                        metrics_regions_020 = create_empty_metrics_dict()

                        metrics_030 = create_empty_metrics_dict()
                        metrics_fire_030 = create_empty_metrics_dict()
                        metrics_regions_030 = create_empty_metrics_dict()

                        metrics_035 = create_empty_metrics_dict()
                        metrics_fire_035 = create_empty_metrics_dict()
                        metrics_regions_035 = create_empty_metrics_dict()

                        metrics_040 = create_empty_metrics_dict()
                        metrics_fire_040 = create_empty_metrics_dict()
                        metrics_regions_040 = create_empty_metrics_dict()

                        metrics_045 = create_empty_metrics_dict()
                        metrics_fire_045 = create_empty_metrics_dict()
                        metrics_regions_045 = create_empty_metrics_dict()

                        metrics_050 = create_empty_metrics_dict()
                        metrics_fire_050 = create_empty_metrics_dict()
                        metrics_regions_050 = create_empty_metrics_dict()

                        metrics_051 = create_empty_metrics_dict()
                        metrics_fire_051 = create_empty_metrics_dict()
                        metrics_regions_051 = create_empty_metrics_dict()

                        metrics_053 = create_empty_metrics_dict()
                        metrics_fire_053 = create_empty_metrics_dict()
                        metrics_regions_053 = create_empty_metrics_dict()

                        metrics_055 = create_empty_metrics_dict()
                        metrics_fire_055 = create_empty_metrics_dict()
                        metrics_regions_055 = create_empty_metrics_dict()

                        metrics_060 = create_empty_metrics_dict()
                        metrics_fire_060 = create_empty_metrics_dict()
                        metrics_regions_060 = create_empty_metrics_dict()

                        metrics_070 = create_empty_metrics_dict()
                        metrics_fire_070 = create_empty_metrics_dict()
                        metrics_regions_070 = create_empty_metrics_dict()

                    for i in range(0, val_batch_nums):
                        # get batched windows for inputs, targets, and fire region masks
                        label_batch = X_val[i * val_batch_size : (i+1) * val_batch_size] # TODO: allow to test the entire validation set at once in gpu implementation (stuck at max of 50 due to memory issues!)
                        label_batch_windows = batched_indexed_windows(label_batch, data, labels, num_training_days, prediction_day,  include_masks=True, masks_full=masks)
                        test_inputs, test_targets, test_regions = label_batch_windows[0], label_batch_windows[1], label_batch_windows[2]
                        test_predictions = model(test_inputs)

                        # TODO CONDENSE THIS (please, I cry)
                        # calculate all of the metrics (its a lot)
                        metrics = calculate_metrics(test_predictions, test_targets, batch_flat_shape_val, default_threshold_value, metrics)
                        metrics_regions = calculate_metrics(test_predictions * test_regions, test_targets, batch_flat_shape_val, default_threshold_value, metrics_regions)
                        metrics_fire = calculate_metrics(test_predictions * test_targets, test_targets, batch_flat_shape_val, default_threshold_value, metrics_fire)

                        if log_thresholding:
                            metrics_010 = calculate_metrics(test_predictions, test_targets, batch_flat_shape_val, 0.10, metrics_010)
                            metrics_regions_010 = calculate_metrics(test_predictions * test_regions, test_targets, batch_flat_shape_val, 0.10, metrics_regions_010)
                            metrics_fire_010 = calculate_metrics(test_predictions * test_targets, test_targets, batch_flat_shape_val, 0.10, metrics_fire_010)

                            metrics_015 = calculate_metrics(test_predictions, test_targets, batch_flat_shape_val, 0.15, metrics_015)
                            metrics_regions_015 = calculate_metrics(test_predictions * test_regions, test_targets, batch_flat_shape_val, 0.15, metrics_regions_015)
                            metrics_fire_015 = calculate_metrics(test_predictions * test_targets, test_targets, batch_flat_shape_val, 0.15, metrics_fire_015)

                            metrics_020 = calculate_metrics(test_predictions, test_targets, batch_flat_shape_val, 0.20, metrics_020)
                            metrics_regions_020 = calculate_metrics(test_predictions * test_regions, test_targets, batch_flat_shape_val, 0.20, metrics_regions_020)
                            metrics_fire_020 = calculate_metrics(test_predictions * test_targets, test_targets, batch_flat_shape_val, 0.20, metrics_fire_020)

                            metrics_030 = calculate_metrics(test_predictions, test_targets, batch_flat_shape_val, 0.30, metrics_030)
                            metrics_regions_030 = calculate_metrics(test_predictions * test_regions, test_targets, batch_flat_shape_val, 0.30, metrics_regions_030)
                            metrics_fire_030 = calculate_metrics(test_predictions * test_targets, test_targets, batch_flat_shape_val, 0.30, metrics_fire_030)

                            metrics_035 = calculate_metrics(test_predictions, test_targets, batch_flat_shape_val, 0.35, metrics_035)
                            metrics_regions_035 = calculate_metrics(test_predictions * test_regions, test_targets, batch_flat_shape_val, 0.35, metrics_regions_035)
                            metrics_fire_035 = calculate_metrics(test_predictions * test_targets, test_targets, batch_flat_shape_val, 0.35, metrics_fire_035)

                            metrics_040 = calculate_metrics(test_predictions, test_targets, batch_flat_shape_val, 0.40, metrics_040)
                            metrics_regions_040 = calculate_metrics(test_predictions * test_regions, test_targets, batch_flat_shape_val, 0.40, metrics_regions_040)
                            metrics_fire_040 = calculate_metrics(test_predictions * test_targets, test_targets, batch_flat_shape_val, 0.40, metrics_fire_040)

                            metrics_045 = calculate_metrics(test_predictions, test_targets, batch_flat_shape_val, 0.45, metrics_045)
                            metrics_regions_045 = calculate_metrics(test_predictions * test_regions, test_targets, batch_flat_shape_val, 0.45, metrics_regions_045)
                            metrics_fire_045 = calculate_metrics(test_predictions * test_targets, test_targets, batch_flat_shape_val, 0.45, metrics_fire_045)

                            metrics_050 = calculate_metrics(test_predictions, test_targets, batch_flat_shape_val, 0.50, metrics_050)
                            metrics_regions_050 = calculate_metrics(test_predictions * test_regions, test_targets, batch_flat_shape_val, 0.50, metrics_regions_050)
                            metrics_fire_050 = calculate_metrics(test_predictions * test_targets, test_targets, batch_flat_shape_val, 0.50, metrics_fire_050)

                            metrics_051 = calculate_metrics(test_predictions, test_targets, batch_flat_shape_val, 0.51, metrics_051)
                            metrics_regions_051 = calculate_metrics(test_predictions * test_regions, test_targets, batch_flat_shape_val, 0.51, metrics_regions_051)
                            metrics_fire_051 = calculate_metrics(test_predictions * test_targets, test_targets, batch_flat_shape_val, 0.51, metrics_fire_051)

                            metrics_053 = calculate_metrics(test_predictions, test_targets, batch_flat_shape_val, 0.53, metrics_053)
                            metrics_regions_053 = calculate_metrics(test_predictions * test_regions, test_targets, batch_flat_shape_val, 0.53, metrics_regions_053)
                            metrics_fire_053 = calculate_metrics(test_predictions * test_targets, test_targets, batch_flat_shape_val, 0.53, metrics_fire_053)

                            metrics_055 = calculate_metrics(test_predictions, test_targets, batch_flat_shape_val, 0.55, metrics_055)
                            metrics_regions_055 = calculate_metrics(test_predictions * test_regions, test_targets, batch_flat_shape_val, 0.55, metrics_regions_055)
                            metrics_fire_055 = calculate_metrics(test_predictions * test_targets, test_targets, batch_flat_shape_val, 0.55, metrics_fire_055)

                            metrics_060 = calculate_metrics(test_predictions, test_targets, batch_flat_shape_val, 0.60, metrics_060)
                            metrics_regions_060 = calculate_metrics(test_predictions * test_regions, test_targets, batch_flat_shape_val, 0.60, metrics_regions_060)
                            metrics_fire_060 = calculate_metrics(test_predictions * test_targets, test_targets, batch_flat_shape_val, 0.60, metrics_fire_060)

                            metrics_070 = calculate_metrics(test_predictions, test_targets, batch_flat_shape_val, 0.70, metrics_070)
                            metrics_regions_070 = calculate_metrics(test_predictions * test_regions, test_targets, batch_flat_shape_val, 0.70, metrics_regions_070)
                            metrics_fire_070 = calculate_metrics(test_predictions * test_targets, test_targets, batch_flat_shape_val, 0.70, metrics_fire_070)

                        # calculate losses
                        full_test_loss = bce_loss(test_predictions, test_targets)
                        fire_test_loss = bce_loss(test_predictions * test_targets, test_targets)
                        fire_test_region_loss = bce_loss(test_predictions*test_regions, test_targets)
                        test_loss = (full_test_loss * 0.2) + (fire_test_region_loss * 0.8)

                        # update total losses
                        val_scaled_loss += test_loss
                        val_fire_region_loss += fire_test_region_loss
                        val_fire_loss += fire_test_loss
                        val_full_loss += full_test_loss

                    # print validation metrics for full set
                    print(f"Validation Batch Loss: Batch Num {batch_num}, Loss: {val_scaled_loss}")
                    print(f"Validation Batch Accuracy: {metrics['validation_accuracy']}, Precision: {metrics['validation_precision']}, Recall: {metrics['validation_recall']}, F1: {metrics['validation_f1']}")

                    # create metrics dictionary for tensorboard
                    metrics_dict = {"training_scaled_bce_loss": loss.item(),
                                    "training_full_bce_loss": full_loss.item(),
                                    "training_fire_loss": fire_loss.item(),
                                    "training_fire_region_loss": region_loss.item(),
                                    "validation_scaled_bce_loss": val_scaled_loss.item(),
                                    "validation_fire_bce_loss": val_fire_loss.item(),
                                    "validation_fire_region_bce_loss": val_fire_region_loss.item(),
                                    "validation_full_region_bce_loss": val_full_loss.item(),
                                    "train_accuracy_0515": train_metrics_dict["accuracy"],
                                    "train_precision_0515": train_metrics_dict["precision"],
                                    "train_recall_0515": train_metrics_dict["recall"],
                                    "train_f1_0515": train_metrics_dict["f1"]}

                    # save metrics to tensorboard
                    tb_optimizer(writer=writer, losses_dict=metrics_dict, step=batch_num)
                    tb_optimizer(writer=writer, losses_dict=metrics, step=batch_num)
                    tb_optimizer(writer=writer_fire, losses_dict=metrics_fire, step=batch_num)
                    tb_optimizer(writer=writer_region, losses_dict=metrics_regions, step=batch_num)

                    if log_thresholding:
                        tb_optimizer(writer=writer_010, losses_dict=metrics_010, step=batch_num)
                        tb_optimizer(writer=writer_015, losses_dict=metrics_015, step=batch_num)
                        tb_optimizer(writer=writer_020, losses_dict=metrics_020, step=batch_num)
                        tb_optimizer(writer=writer_030, losses_dict=metrics_030, step=batch_num)
                        tb_optimizer(writer=writer_035, losses_dict=metrics_035, step=batch_num)
                        tb_optimizer(writer=writer_040, losses_dict=metrics_040, step=batch_num)
                        tb_optimizer(writer=writer_045, losses_dict=metrics_045, step=batch_num)
                        tb_optimizer(writer=writer_050, losses_dict=metrics_050, step=batch_num)
                        tb_optimizer(writer=writer_051, losses_dict=metrics_051, step=batch_num)
                        tb_optimizer(writer=writer_053, losses_dict=metrics_053, step=batch_num)
                        tb_optimizer(writer=writer_055, losses_dict=metrics_055, step=batch_num)
                        tb_optimizer(writer=writer_060, losses_dict=metrics_060, step=batch_num)
                        tb_optimizer(writer=writer_070, losses_dict=metrics_070, step=batch_num)

                        tb_optimizer(writer=writer_fire_010, losses_dict=metrics_fire_010, step=batch_num)
                        tb_optimizer(writer=writer_fire_015, losses_dict=metrics_fire_015, step=batch_num)
                        tb_optimizer(writer=writer_fire_020, losses_dict=metrics_fire_020, step=batch_num)
                        tb_optimizer(writer=writer_fire_030, losses_dict=metrics_fire_030, step=batch_num)
                        tb_optimizer(writer=writer_fire_035, losses_dict=metrics_fire_035, step=batch_num)
                        tb_optimizer(writer=writer_fire_040, losses_dict=metrics_fire_040, step=batch_num)
                        tb_optimizer(writer=writer_fire_045, losses_dict=metrics_fire_045, step=batch_num)
                        tb_optimizer(writer=writer_fire_050, losses_dict=metrics_fire_050, step=batch_num)
                        tb_optimizer(writer=writer_fire_051, losses_dict=metrics_fire_051, step=batch_num)
                        tb_optimizer(writer=writer_fire_053, losses_dict=metrics_fire_053, step=batch_num)
                        tb_optimizer(writer=writer_fire_055, losses_dict=metrics_fire_055, step=batch_num)
                        tb_optimizer(writer=writer_fire_060, losses_dict=metrics_fire_060, step=batch_num)
                        tb_optimizer(writer=writer_fire_070, losses_dict=metrics_fire_070, step=batch_num)

                        tb_optimizer(writer=writer_region_010, losses_dict=metrics_regions_010, step=batch_num)
                        tb_optimizer(writer=writer_region_015, losses_dict=metrics_regions_015, step=batch_num)
                        tb_optimizer(writer=writer_region_020, losses_dict=metrics_regions_020, step=batch_num)
                        tb_optimizer(writer=writer_region_030, losses_dict=metrics_regions_030, step=batch_num)
                        tb_optimizer(writer=writer_region_035, losses_dict=metrics_regions_035, step=batch_num)
                        tb_optimizer(writer=writer_region_040, losses_dict=metrics_regions_040, step=batch_num)
                        tb_optimizer(writer=writer_region_045, losses_dict=metrics_regions_045, step=batch_num)
                        tb_optimizer(writer=writer_region_050, losses_dict=metrics_regions_050, step=batch_num)
                        tb_optimizer(writer=writer_region_051, losses_dict=metrics_regions_051, step=batch_num)
                        tb_optimizer(writer=writer_region_053, losses_dict=metrics_regions_053, step=batch_num)
                        tb_optimizer(writer=writer_region_055, losses_dict=metrics_regions_055, step=batch_num)
                        tb_optimizer(writer=writer_region_060, losses_dict=metrics_regions_060, step=batch_num)
                        tb_optimizer(writer=writer_region_070, losses_dict=metrics_regions_070, step=batch_num)
                        # tb_optimizer(writer=writer, losses_dict=threshold_test_mets, step=batch_num)

                    # save model checkpoint every 150 batches (1500 samples)
                    if (batch_num % 150) == 0:
                        checkpoint = {
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()}#,  # If you have an optimizer
                            # Add other relevant information like epoch, loss, etc.}
                        torch.save(checkpoint, f'{checkpoint_dir}/checkpoint_epoch_{epoch}_batch_{batch_num}.pth')
                        logging.info(
                            f"Model checkpoint for epoch {epoch} saved to: {checkpoint_dir}/checkpoint_epoch_{epoch}_batch_{batch_num}.pth")

            loss.backward()
            batch_num += 1

            optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        # save model state dictionary
        torch.save(model.state_dict(), f'{checkpoint_dir}/model_epoch_{epoch}.pth')
        logging.info(f"Model state dictionary for epoch {epoch} saved to: {checkpoint_dir}/model_epoch_{epoch}.pth")

        # save entire model (not recommended for prod)
        checkpoint = {'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict()}
        torch.save(checkpoint, f'{checkpoint_dir}/checkpoint_epoch_{epoch}.pth')
        logging.info(f"Model checkpoint for epoch {epoch} saved to: {checkpoint_dir}/checkpoint_epoch_{epoch}.pth")

    # generate predictions for entire test set
    if generate_predictions:
        # output_df = pd.DataFrame()
        # # columns are date lat long is_fire_day risk_score

        with torch.no_grad():
            optimizer.zero_grad()
            for i in X_test:
                # set batch to just one index
                batch = [i]
                # get windows for inputs, targets, and fire region masks
                batch_windows = batched_indexed_windows(batch, data, labels, num_training_days, prediction_day, include_masks=True, masks_full=masks)
                inputs, targets, regions = batch_windows[0], batch_windows[1], batch_windows[2]
                # forward pass, generate prediction from model
                outputs = model(inputs)

                # plot output risk scores and save figure
                plot_target_vs_predictions(batch, outputs.detach().cpu().numpy(), targets.detach().cpu().numpy(), pred_batch_size=1, batch_num=batch_num, cmap=cmap_default, norm=norm_default, save_images=True, root_dir=f'./test_outputs_epoch_{epoch}/', prediction_day=prediction_day)



app()
#
# # COMMENTING OUT BECAUSE USING TYPER
# if __name__ == "__main__":
#     main()
