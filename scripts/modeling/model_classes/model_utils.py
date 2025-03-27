from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def cross_validate_model(model_obj, X, y, n_splits=5, random_state=42):
    """
    Perform K-Fold cross-validation on the given model.
    Args:
        model_obj: An object with .train() and .evaluate() methods (inherits from BaseModel).
        X (np.ndarray): Feature matrix as a NumPy array.
        y (np.ndarray): Target array as NumPy.
        n_splits (int): Number of folds for cross-validation.
        random_state (int): For reproducible splits.

    Returns:
        metrics_avg (dict): Dictionary of average evaluation metrics across folds.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    metrics_list = []
    fold = 1
    
    for train_index, test_index in kf.split(X):
        print(f"Processing fold {fold}...")

        # X and y are NumPy arrays, so we use standard slicing instead of iloc
        X_train_fold = X[train_index]
        X_test_fold = X[test_index]
        y_train_fold = y[train_index]
        y_test_fold = y[test_index]

        # Train the model on this fold
        model_obj.train(X_train_fold, y_train_fold)
        
        # Evaluate on this fold
        metrics = model_obj.evaluate(X_test_fold, y_test_fold)
        metrics_list.append(metrics)
        
        print(f"Fold {fold} metrics: {metrics}\n")
        fold += 1

    # Average metrics over all folds
    metrics_avg = {}
    if metrics_list:
        # assume each metrics dict has same keys
        keys = metrics_list[0].keys()
        for key in keys:
            metrics_avg[key] = np.mean([m[key] for m in metrics_list])

    print("Average Cross-Validation Metrics:")
    print(metrics_avg)
    return metrics_avg
