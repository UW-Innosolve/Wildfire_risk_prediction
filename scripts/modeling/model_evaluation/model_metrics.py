

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class BaseModel:
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
