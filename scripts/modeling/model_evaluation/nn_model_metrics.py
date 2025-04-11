import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score, roc_auc_score


# class BaseModel:
#     """
#     Base model class that provides a generic evaluate method.
#     All model classes will inherit from this to avoid repeating evaluation code.
#     """
# TODO: troubleshoot roc_auc_score
def evaluate(predictions, targets, flat_shape, threshold_value=0.515):
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
    # try:
    #     # Some models provide predict_proba for probability estimates.
    #     probs = model.predict_proba(X_test)[:, 1]
    # except AttributeError:
    #     # If predict_proba isn't available, use decision_function as a fallback.
    #     probs = model.decision_function(X_test)

    # flatten arrays
    predictions_flat = predictions.detach().cpu().numpy().reshape(flat_shape)
    targets_flat = targets.detach().cpu().numpy().reshape(flat_shape)

    # threshold predictions to give 1 and 0
    thresholded_predictions_flat = (predictions_flat > threshold_value).astype(int)

    # Compute metrics using scikit-learn's functions.
    metrics = {
        "accuracy": accuracy_score(targets_flat.astype(int), thresholded_predictions_flat),
        "precision": precision_score(targets_flat.astype(int), thresholded_predictions_flat),
        "recall": recall_score(targets_flat.astype(int), thresholded_predictions_flat),
        "f1": f1_score(targets_flat.astype(int), thresholded_predictions_flat),
        # "roc_auc": roc_auc_score(targets_flat, predictions_flat)
    }
    return metrics

def evaluate_individuals(predictions, targets, flat_shape, threshold_value=0.515):
    """
    Evaluate the model using standard metrics:
      - Accuracy: Overall correctness.
      - Precision: Ratio of true positive predictions to total positive predictions.
      - Recall: Ratio of true positive predictions to actual positives.
      - F1 Score: Harmonic mean of precision and recall.
      - ROC-AUC: Measure of separability.

    Returns:
        accuracy, precision, recall, f1_score
    """
    # Predict class labels using the model's predict method.
    # try:
    #     # Some models provide predict_proba for probability estimates.
    #     probs = model.predict_proba(X_test)[:, 1]
    # except AttributeError:
    #     # If predict_proba isn't available, use decision_function as a fallback.
    #     probs = model.decision_function(X_test)

    # flatten arrays
    predictions_flat = predictions.detach().cpu().numpy().reshape(flat_shape)
    targets_flat = targets.detach().cpu().numpy().reshape(flat_shape)

    # threshold predictions to give 1 and 0
    thresholded_predictions_flat = (predictions_flat > threshold_value).astype(int)
    if targets_flat.sum() == 0:
        print('No fire areas present in batch')
    if thresholded_predictions_flat.sum() == 0:
        print(f"No fire areas predicted in batch using threshold {threshold_value}")

    accuracy = accuracy_score(targets_flat, thresholded_predictions_flat)
    precision = precision_score(targets_flat.astype(int), thresholded_predictions_flat)
    recall = recall_score(targets_flat.astype(int), thresholded_predictions_flat)
    f1 = f1_score(targets_flat.astype(int), thresholded_predictions_flat)

    return accuracy, precision, recall, f1