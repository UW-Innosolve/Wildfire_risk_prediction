import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import pandas as pd
import tensorboard as tb


# input data has shape (16, )
#       16 training samples
#       10 days per sample
#       31 parameters per day
#       37x34 array per parameter


## FROM GEMINI
class LSTM_3D(nn.Module):
    def __init__(self, input_channels, hidden_size, dropout_rate):
        super(LSTM_3D, self).__init__()
        self.lstm1 = nn.LSTM(input_size=input_channels, hidden_size=hidden_size, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.lstm2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(hidden_size, 1)# Outputting a single value per timestep
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, 10, 31, 37, 34]
        Returns:
            output: Output tensor of shape [batch_size, 37, 34]
        """
        batch_size, time_steps, depth, height, width = x.size()

        # Reshape the input to [batch_size * height * width, time_steps, depth]
        x = x.permute(0, 3, 4, 1, 2).contiguous()  # [batch_size, height, width, time_steps, depth]
        x = x.view(-1, time_steps, depth)  # [batch_size * height * width, time_steps, depth]

        # LSTM layers
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)

        # Take the last timestep's output
        out = out[:, -1, :]  # [batch_size * height * width, hidden_size]

        # Linear layer to get a single output per timestep
        out = self.linear(out)  # [batch_size * height * width, 1]

        # take sigmoid of linear layer to get output for binary classification
        out = self.sigmoid(out)

        # Reshape back to [batch_size, height, width]
        out = out.view(batch_size, height, width)  # [batch_size, height, width]

        return out


# # Example usage:
# batch_size = 14
# time_steps = 10
# depth = 42
# height = 37
# width = 34
# hidden_size = 64
# dropout_rate = 0.2
#
# # Create a dummy input tensor
# input_tensor = torch.randn(batch_size, time_steps, depth, height, width)
#
# # Create the model
# model = LSTM_3D(input_channels=depth, hidden_size=hidden_size, dropout_rate=dropout_rate)
#
# # Forward pass
# output = model(input_tensor)
#
# # Print the output shape
# print(output.shape)  # Should be [2, 37, 34]

