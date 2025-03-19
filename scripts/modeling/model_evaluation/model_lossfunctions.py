import torch.nn as nn


def binary_cross_entropy_loss(output, target):
    """
    Calculates the binary cross-entropy loss.

    Args:
        output (torch.Tensor): Output tensor of probabilities (after sigmoid), shape [batch_size, ...].
        target (torch.Tensor): Target tensor of binary values, shape [batch_size, ...].

    Returns:
        torch.Tensor: The binary cross-entropy loss.
    """

    criterion = nn.BCELoss()  # Use BCELoss

    loss = criterion(output, target)

    return loss