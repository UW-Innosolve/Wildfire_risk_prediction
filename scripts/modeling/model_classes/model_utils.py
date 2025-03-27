from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import pandas as pd

def cross_validate_model(model, X, y, n_splits=5, random_state=42):
    """
    Perform K-Fold cross-validation on the given model.
    Args:
        model: An object with train() and evaluate() methods (inherits from BaseModel).
        X: pandas DataFrame of features.
        y: pandas Series or NumPy array of target.
        n_splits: Number of folds for cross-validation.
        random_state: For reproducibility.
    Returns:
        metrics_avg: Dictionary of average evaluation metrics across folds.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    metrics_list = []
    fold = 1
    
    for train_index, test_index in kf.split(X):
        print(f"Processing fold {fold}...")

        # Indexing X (Pandas DataFrame)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        
        # Indexing y: Handle both Pandas Series and NumPy array
        if isinstance(y, pd.Series):  # If y is a Pandas Series (before ravel)
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        else:  # If y is a NumPy array (after ravel)
            y_train, y_test = y[train_index], y[test_index]

        # Train the model on this fold
        model.train(X_train, y_train)
        
        # Evaluate and collect metrics
        metrics = model.evaluate(X_test, y_test)
        metrics_list.append(metrics)
        
        print(f"Fold {fold} metrics: {metrics}\n")
        fold += 1

    # Average metrics over all folds
    metrics_avg = {key: np.mean([m[key] for m in metrics_list]) for key in metrics_list[0]}
    
    print("Average Cross-Validation Metrics:")
    print(metrics_avg)
    return metrics_avg