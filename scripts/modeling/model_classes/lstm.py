# import torch
import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


class BaseModel(nn.Module):
    """
    Base model class that provides a generic evaluate method.
    All model classes will inherit from this to avoid repeating evaluation code.
    """

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model using standard metrics:
          - Accuracy: Overall correctness.
          - Precision: Ratio of true positive predictions to total positive predictions.
          - Recall: Ratio of true positive predictions to actual positives.
          - F1 Score: Harmonic mean of precision and recall.
          - ROC-AUC: Measure of separability.

        Returns:
            metrics (dict): A dictionary containing evaluation metrics.
        """
        # Predict class labels using the model's predict method.
        predictions = self.model.predict(X_test)
        try:
            # Some models provide predict_proba for probability estimates.
            probs = self.model.predict_proba(X_test)[:, 1]
        except AttributeError:
            # If predict_proba isn't available, use decision_function as a fallback.
            probs = self.model.decision_function(X_test)

        # Compute metrics using scikit-learn's functions.
        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions),
            "recall": recall_score(y_test, predictions),
            "f1_score": f1_score(y_test, predictions),
            "roc_auc": roc_auc_score(y_test, probs)
        }
        return metrics


class LSTM_3D(BaseModel):
    """
    Class for 3-Dimensional (2D + time) LSTM with 2 hidden layers.
    """
    def __init__(self, input_channels, hidden_size, dropout_rate):
        """
        Description: instantiates LSTM model

        Parameters:
            input_channels: int
                Number of features in input data
            hidden_size: int
                Size of each hidden layer.
            dropout_rate: float
                Fraction of dropout to be applied
                Should be between 0 and 1

        Returns:
            Instantiated LSTM_3D model class.

        Notes:
            Both hidden layers have the same size in this model.
        """
        super(LSTM_3D, self).__init__()
        self.lstm1 = nn.LSTM(input_size=input_channels, hidden_size=hidden_size, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.lstm2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(hidden_size, 1) # outputs a single value per timestep
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Description: batched forward pass, generates batched output prediction.

        Parameters:
            x: torch.Tensor
                Input tensor of shape [batch_size, num_training_days, features, 37, 34]
        Returns:
            output: torch.Tensor
                Output tensor (prediction after forward pass) of shape [batch_size, 37, 34]
        """
        batch_size, time_steps, depth, height, width = x.size()

        # rearrange input to [batch_size * height * width, time_steps, depth]
        x = x.permute(0, 3, 4, 1, 2).contiguous()  # [batch_size, height, width, time_steps, depth]
        x = x.view(-1, time_steps, depth)  # [batch_size * height * width, time_steps, depth]

        # LSTM layers
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)

        # take output of last timestep
        out = out[:, -1, :]  # [batch_size * height * width, hidden_size]

        # linear layer to get a single output per timestep
        out = self.linear(out)  # [batch_size * height * width, 1]

        # take sigmoid of linear layer to get output for binary classification
        # outputs between 0 and 1
        out = self.sigmoid(out)

        # reshape back to [batch_size, height, width]
        out = out.view(batch_size, height, width)  # [batch_size, height, width]

        return out


class LSTM_3D_3layers(BaseModel):
    def __init__(self, input_channels, hidden_size, dropout_rate):
        super(LSTM_3D, self).__init__()
        self.lstm1 = nn.LSTM(input_size=input_channels, hidden_size=hidden_size, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.lstm2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.lstm3 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(hidden_size, 1)# Outputting a single value per timestep
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, training_days, 31, 37, 34]
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


