from scripts.modeling.model_evaluation.model_lossfunctions import binary_cross_entropy_loss as bce_loss
from scripts.modeling.model_classes.lstm import LSTM_3D
from scripts.modeling.data_preprocessing.dataloading import dataloader
from scripts.modeling.data_preprocessing.windowing import create_windows

import torch.optim as optim
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


# TODO create a training_parameters json or something similar to make tracking easier?
def train_lstm(dataset, labels, training_parameters):
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

    # generate windowed data
    data, labels = dataloader()
    windowed_data, windowed_labels = create_windows(data, labels, num_training_days, prediction_day)

    # Split the data; apply SMOTE for balancing minority class (fire days).
    logging.info("Splitting data into training and test sets and applying SMOTE for balancing...")
    # TODO complete train test splitting
    X_train, X_test, y_train, y_test = [], [], [], []

    # create model
    model = LSTM_3D()

    # set
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    input_data = torch.randn(batch_size, 3, 5) # example input

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(input_data.view(batch_size * 3, 5)).view(batch_size, 3, 1)  # forward pass
        outputs = outputs.squeeze(2)  # remove extra dimension
        loss = bce_loss(outputs, target)
        loss.backward()

        optimizer.step()
        if (epoch) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


if __name__ == "__nn_main__":
    train_lstm()
