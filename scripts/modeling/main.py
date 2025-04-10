from model_evaluation.model_lossfunctions import binary_cross_entropy_loss as bce_loss
from model_classes.lstm import LSTM_3D
from data_preprocessing.windowing import batched_indexed_windows, reshape_data
from model_evaluation.nn_model_metrics import evaluate, evaluate_individuals


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
# TODO check min max values of predictions at various stages
# TODO check min max values of just fire locations, and also get a loss for just those locations (see how well its actually doing)
# TODO ask the model to predict classes AND probabilities
# TODO create a threshold function for predictions, current threshold set to 0.515 (could be way off idk)
def main(training_parameters={"batch_size": 10,
                              "num_epochs": 20,
                              "learning_rate": 0.05,
                              "num_training_days": 14,
                              "prediction_day":5,
                              "hidden_size": 64,
                              "experiment_name":"testwrite_tb_b",
                              "test_range": (2008),
                              "train_range": (2006, 2007)},
         # rawdata_path='/home/tvujovic/scratch/firebird/processed_data.csv',
         rawdata_path='/Users/teodoravujovic/Desktop/code/firebird/processed_data.csv',
         device_set='cuda',
         include_masks=True, # TODO make sure this is included everywhere for modularity, make sure it doesnt slow things down too much
         mask_size=2,
         threshold_value=0.515):
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
    rawdata_df = pd.read_csv(rawdata_path)[:1377510] #.to(device)
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
    reshaped_data, reshaped_labels, reshaped_masks = reshape_data(rawdata_df, features, target_column, device_set, include_masks=include_masks)
    # logging
    logging.info(f"Successfully reshaped all features")

    # remove after converting reshape function to torch
    data = torch.Tensor(reshaped_data).to(device)
    labels = torch.Tensor(reshaped_labels).to(device)
    masks = torch.Tensor(reshaped_masks).to(device)

    # TODO: make it modular which subdirectories to create
    # set tensorboard writer directory and subdirectories
    writer = SummaryWriter(log_dir=f"{checkpoint_dir}threshold_{threshold_value}")
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

    # set flattened array size for one batch (training and validation separately)
    # TODO fix so this is modular
    batch_flat_shape = 12580
    batch_flat_shape_val = 37740

    # set a list of possible threshold values to test
    threshold_value_testlist = [0.45, 0.5, 0.51, 0.53, 0.55, 0.6, 0.7]

    # # TODO: fix so its not hardcoded
    # # create tensorboard writers for all other test threshold values
    # writer_045 = SummaryWriter(log_dir=f"{checkpoint_dir}threshold_045")
    # writer_050 = SummaryWriter(log_dir=f"{checkpoint_dir}threshold_050")
    # writer_051 = SummaryWriter(log_dir=f"{checkpoint_dir}threshold_051")
    # writer_053 = SummaryWriter(log_dir=f"{checkpoint_dir}threshold_053")
    # writer_055 = SummaryWriter(log_dir=f"{checkpoint_dir}threshold_055")
    # writer_060 = SummaryWriter(log_dir=f"{checkpoint_dir}threshold_060")
    # writer_070 = SummaryWriter(log_dir=f"{checkpoint_dir}threshold_070")

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
                    train_metrics_dict = evaluate(outputs, targets, batch_flat_shape, threshold_value=threshold_value)
                    print(f"Accuracy: {train_metrics_dict['accuracy']}, Precision: {train_metrics_dict['precision']}, Recall: {train_metrics_dict['recall']}, F1: {train_metrics_dict['f1']}")

                    # set total validation losses and metrics to 0, we will average over the entire validation set later
                    val_scaled_loss = 0
                    val_full_loss = 0
                    val_fire_loss = 0
                    val_fire_region_loss = 0
                    val_accuracy = 0
                    val_precision = 0
                    val_recall = 0
                    val_f1 = 0
                    val_region_accuracy = 0
                    val_region_precision = 0
                    val_region_recall = 0
                    val_region_f1 = 0
                    val_fire_accuracy = 0
                    val_fire_precision = 0
                    val_fire_recall = 0
                    val_fire_f1 = 0
                    max_value = 0
                    min_value = 0
                    avg_max_value = 0
                    avg_min_value = 0

                    # validation metrics for TESTING THRESHOLDS (won't be here forever)
                    # TODO figure out a more elegant way to not hardcode this
                    # update length of list of 0 depending on number of test thresholds
                    val_mets = np.zeros((4, len(threshold_value_testlist)))
                    val_region_mets = np.zeros((4, len(threshold_value_testlist)))
                    val_fire_mets = np.zeros((4, len(threshold_value_testlist)))

                    # random shuffle the order of the validation set
                    # TODO: does this matter since we aren't learning anyways?
                    np.random.shuffle(X_val)

                    for i in range(0, val_batch_nums):
                        # get batched windows for inputs, targets, and fire region masks
                        label_batch = X_val[i * val_batch_size : (i+1) * val_batch_size] # TODO: allow to test the entire validation set at once in gpu implementation (stuck at max of 50 due to memory issues!)
                        label_batch_windows = batched_indexed_windows(label_batch, data, labels, num_training_days, prediction_day,  include_masks=True, masks_full=masks)
                        test_inputs, test_targets, test_regions = label_batch_windows[0], label_batch_windows[1], label_batch_windows[2]
                        test_predictions = model(test_inputs)

                        # calculate validation metrics for this batch
                        batch_val_accuracy, batch_val_precision, batch_val_recall, batch_val_f1 = evaluate_individuals(test_predictions, test_targets, batch_flat_shape_val, threshold_value=threshold_value)
                        batch_val_region_accuracy, batch_val_region_precision, batch_val_region_recall, batch_val_region_f1 = evaluate_individuals(test_predictions * test_regions, test_targets, batch_flat_shape_val, threshold_value=threshold_value)
                        batch_val_fire_accuracy, batch_val_fire_precision, batch_val_fire_recall, batch_val_fire_f1 = evaluate_individuals(
                            test_predictions * test_targets, test_targets, batch_flat_shape_val,
                            threshold_value=threshold_value)

                        # calculate validation metrics for test threshold values
                        # TODO: un-hardcode this :), hard-coded for 5 different threshold values
                        # THIS SUCKS BUT I NEED IT (sorry)
                        for j in range(len(threshold_value_testlist)):
                            temp_acc, temp_prec, temp_rec, temp_f1 = evaluate_individuals(test_predictions, test_targets, batch_flat_shape_val, threshold_value=threshold_value_testlist[j])
                            temp_reg_acc, temp_reg_prec, temp_reg_rec, temp_reg_f1 = evaluate_individuals(test_predictions * test_regions, test_targets, batch_flat_shape_val, threshold_value=threshold_value_testlist[j])
                            temp_fire_acc, temp_fire_prec, temp_fire_rec, temp_fire_f1 = evaluate_individuals(test_predictions * test_targets, test_targets, batch_flat_shape_val, threshold_value=threshold_value_testlist[j])

                            val_mets[:, j] += temp_acc, temp_prec, temp_rec, temp_f1
                            val_region_mets[:, j] += temp_reg_acc, temp_reg_prec, temp_reg_rec, temp_reg_f1
                            val_fire_mets[:, j] += temp_fire_acc, temp_fire_prec, temp_fire_rec, temp_fire_f1

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
                        val_accuracy += batch_val_accuracy
                        val_recall += batch_val_recall
                        val_f1 += batch_val_f1
                        val_precision += batch_val_precision
                        val_region_accuracy += batch_val_region_accuracy
                        val_region_precision += batch_val_region_precision
                        val_region_recall += batch_val_region_recall
                        val_region_f1 += batch_val_region_f1
                        val_fire_accuracy += batch_val_fire_accuracy
                        val_fire_precision += batch_val_fire_precision
                        val_fire_recall += batch_val_fire_recall
                        val_fire_f1 += batch_val_fire_f1

                    # average losses over the full batch
                    val_scaled_loss /= val_batch_nums
                    val_fire_loss /= val_batch_nums
                    val_fire_region_loss /= val_batch_nums
                    val_full_loss /= val_batch_nums
                    val_accuracy /= val_batch_nums
                    val_recall /= val_batch_nums
                    val_f1 /= val_batch_nums
                    val_precision /= val_batch_nums
                    val_region_accuracy /= val_batch_nums
                    val_region_precision /= val_batch_nums
                    val_region_recall /= val_batch_nums
                    val_region_f1 /= val_batch_nums

                    # average threshold testing metrics over full batch
                    val_mets /= val_batch_nums
                    val_fire_mets /= val_batch_nums
                    val_region_mets /= val_batch_nums

                    # print validation metrics for full set
                    print(f"Validation Batch Loss: Batch Num {batch_num}, Loss: {val_scaled_loss}")
                    print(f"Validation Batch Accuracy: {val_accuracy}, Precision: {val_precision}, Recall: {val_recall}, F1: {val_f1}")

                    # create metrics dictionary for tensorboard
                    metrics_dict = {"training_scaled_bce_loss": loss.item(),
                                    "training_full_bce_loss": full_loss.item(),
                                    "training_fire_loss": fire_loss.item(),
                                    "training_fire_region_loss": region_loss.item(),
                                    "validation_scaled_bce_loss": val_scaled_loss.item(),
                                    "validation_fire_region_bce_loss": val_fire_region_loss.item(),
                                    "validation_full_region_bce_loss": val_full_loss.item(),
                                    "train_accuracy": train_metrics_dict["accuracy"],
                                    "train_precision": train_metrics_dict["precision"],
                                    "train_recall": train_metrics_dict["recall"],
                                    "train_f1": train_metrics_dict["f1"],
                                    "validation_accuracy": val_accuracy,
                                    "validation_precision": val_precision,
                                    "validation_recall": val_recall,
                                    "validation_f1": val_f1,
                                    "validation_fire_region_accuracy": val_region_accuracy,
                                    "validation_fire_region_precision": val_region_precision,
                                    "validation_fire_region_recall": val_region_recall,
                                    "validation_fire_region_f1": val_region_f1,
                                    "validation_fire_accuracy": val_fire_accuracy,
                                    "validation_fire_precision": val_fire_precision,
                                    "validation_fire_recall": val_fire_recall,
                                    "validation_fire_f1": val_fire_f1}
                                    # # "roc_auc": test_metrics["roc_auc"]}

                    threshold_test_mets = {}
                    for i in threshold_value_testlist:
                        for j in range(len(threshold_value_testlist)):
                            threshold_test_mets[f"validation_accuracy_threshold_{i}"] = val_mets[0, j]
                            threshold_test_mets[f"validation_precision_threshold_{i}"] = val_mets[1, j]
                            threshold_test_mets[f"validation_recall_threshold_{i}"] = val_mets[2, j]
                            threshold_test_mets[f"validation_f1_threshold_{i}"] = val_mets[3, j]

                            threshold_test_mets[f"validation_region_accuracy_threshold_{i}"] = val_region_mets[0, j]
                            threshold_test_mets[f"validation_region_precision_threshold_{i}"] = val_region_mets[1, j]
                            threshold_test_mets[f"validation_region_recall_threshold_{i}"] = val_region_mets[2, j]
                            threshold_test_mets[f"validation_region_f1_threshold_{i}"] = val_region_mets[3, j]

                            threshold_test_mets[f"validation_fire_accuracy_threshold_{i}"] = val_fire_mets[0, j]
                            threshold_test_mets[f"validation_fire_precision_threshold_{i}"] = val_fire_mets[1, j]
                            threshold_test_mets[f"validation_fire_recall_threshold_{i}"] = val_fire_mets[2, j]
                            threshold_test_mets[f"validation_fire_f1_threshold_{i}"] = val_fire_mets[3, j]

                    # save metrics to tensorboard
                    tb_optimizer(writer=writer, losses_dict=metrics_dict, step=batch_num)
                    tb_optimizer(writer=writer, losses_dict=threshold_test_mets, step=batch_num)

                    # save model checkpoint every 150 batches (1500 samples)
                    if (batch_num % 150) == 0:
                        checkpoint = {
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()}#,  # If you have an optimizer
                            # Add other relevant information like epoch, loss, etc.}
                        torch.save(checkpoint, f'{checkpoint_dir}/checkpoint_epoch_{epoch}_batch_{batch_num}.pth')
                        logging.info(
                            f"Model checkpoint for epoch {epoch} saved to: {checkpoint_dir}/checkpoint_epoch_{epoch}.pth")

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
