# model_classes/fb_approx_knn.py

from model_evaluation.model_metrics import BaseModel  # assuming BaseModel is a minimal base class
from annoy import AnnoyIndex
import numpy as np
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class ApproxKNNModel(BaseModel):
    """
    An approximate KNN classifier using the Annoy library.
    """
    def __init__(self, n_neighbors=23, n_trees=10, metric='euclidean'):
        """
        Parameters:
          - n_neighbors (int): number of neighbors to retrieve for majority voting
          - n_trees (int): number of trees to build (higher for better accuracy, slower build)
          - metric (str): distance metric ('euclidean', 'angular', etc.)
        """
        self.n_neighbors = n_neighbors
        self.n_trees = n_trees
        self.metric = metric
        self.annoy_index = None
        self.X_train = None
        self.y_train = None
        self.dim = None

    def train(self, X_train, y_train):
        """
        Build the Annoy index for approximate neighbor search.
        X_train: 2D NumPy array (contiguous, numeric), shape (n_samples, n_features)
        y_train: 1D NumPy array, shape (n_samples,)
        """
        logging.info("ApproxKNNModel: Building Annoy index for approximate neighbor search...")
        self.X_train = X_train
        self.y_train = y_train
        self.dim = X_train.shape[1]
        self.annoy_index = AnnoyIndex(self.dim, self.metric)

        # Add each training vector to the Annoy index
        for i in range(len(X_train)):
            self.annoy_index.add_item(i, X_train[i])
        
        # Build the Annoy index with n_trees trees
        self.annoy_index.build(self.n_trees)
        logging.info(f"Annoy index built with {len(X_train)} items, dim={self.dim}, n_trees={self.n_trees}")

    def predict(self, X_test):
        """
        Predict class labels for test samples using majority voting among approximate nearest neighbors.
        X_test: 2D NumPy array, shape (n_test_samples, n_features)
        Returns: 1D NumPy array of predictions.
        """
        preds = []
        for x in X_test:
            nn_indices = self.annoy_index.get_nns_by_vector(x, self.n_neighbors)
            neighbor_labels = self.y_train[nn_indices]
            # Majority vote
            vals, counts = np.unique(neighbor_labels, return_counts=True)
            pred = vals[np.argmax(counts)]
            preds.append(pred)
        return np.array(preds)

    def predict_proba(self, X_test):
        """
        Approximate probability estimation for binary classification.
        Returns an array of shape (n_samples, 2) with probabilities for classes [0, 1].
        """
        probs = []
        for x in X_test:
            nn_indices = self.annoy_index.get_nns_by_vector(x, self.n_neighbors)
            neighbor_labels = self.y_train[nn_indices]
            p = np.mean(neighbor_labels)  # fraction of ones
            probs.append([1 - p, p])
        return np.array(probs)

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.
        Computes accuracy, precision, recall, F1 score, and ROC AUC.
        """
        predictions = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions, zero_division=0),
            'recall': recall_score(y_test, predictions, zero_division=0),
            'f1_score': f1_score(y_test, predictions, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_proba[:, 1]) if len(np.unique(y_test)) > 1 else None
        }
        return metrics
