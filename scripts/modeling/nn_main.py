from scripts.modeling.model_evaluation.model_lossfunctions import binary_cross_entropy_loss as bce_loss
from scripts.modeling.model_classes.lstm import LSTM_3D

import torch.optim as optim
import torch

import tensorboard as tb


# TODO create a training_parameters json or something similar to make tracking easier?
def train_lstm(dataset, labels, training_parameters):
    batch_size = training_parameters['batch_size']
    num_epochs = training_parameters['num_epochs']
    learning_rate = training_parameters['learning_rate']
    features = training_parameters['features']

    data, labels =

    model = LSTM_3D()

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
