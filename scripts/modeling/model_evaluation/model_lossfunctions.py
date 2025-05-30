import torch.nn as nn


def binary_cross_entropy_loss(output, target):
    """

    Description: Calculates and returns the binary cross-entropy loss.

    Parameters:
        output (torch.Tensor):
            Tensor of predicted probabilities output from the model
            Has shape [batch_size, 37, 34] (using default latitude and longitude counts)
        target (torch.Tensor):
            Groundtruth target tensor of binary values
            Has shape [batch_size, 37, 34] (using default latitude and longitude counts)

    Returns:
        torch.Tensor
            Binary cross entropy loss for batch
    """
    # calculate loss
    criterion = nn.BCELoss()
    bceloss = criterion(output, target)

    return bceloss

