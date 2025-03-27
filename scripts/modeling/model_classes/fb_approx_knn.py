from model_evaluation.model_metrics import BaseModel
from annoy import AnnoyIndex
import numpy as np
import logging

class ApproxKNNModel(BaseModel):
    """
    An approximate KNN classifier using the Annoy library.
    """
    def __init__(self, n_neighbors=23, n_trees=10, metric='euclidean'):
        """
        :param n_neighbors: number of neighbors to retrieve for majority voting
        :param n_trees: number of trees Annoy will build (higher = more accuracy, more build time)
        :param metric: 'euclidean' or 'angular' etc. (Annoy supports a few)
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
        Build Annoy index for approximate neighbor search.
        X_train should be a 2D NumPy array, y_train a 1D array.
        """
        logging.info("ApproxKNNModel: Building Annoy index for approximate neighbor search...")
        self.X_train = X_train
        self.y_train = y_train
        self.dim = X_train.shape[1]
        
        # Annoy requires us to specify the dimension and metric
        self.annoy_index = AnnoyIndex(self.dim, self.metric)

        # Add each training vector to the Annoy index
        for i in range(len(X_train)):
            self.annoy_index.add_item(i, X_train[i, :])

        # Build the Annoy index
        # n_trees: bigger => more accurate, slower to build
        self.annoy_index.build(self.n_trees)
        logging.info(f"Annoy index built with {len(X_train)} items, dim={self.dim}, n_trees={self.n_trees}")

    def evaluate(self, X_test, y_test):
        """
        Evaluate on test data using approximate neighbor search for K nearest neighbors.
        """
        # We rely on the base class's evaluate method to do the actual metrics,
        # but we override how we do 'predict'.
        # The base 'evaluate()' calls self.model.predict() or .predict_proba(), but here we define our own predict:
        predictions = self.predict(X_test)
        
        # We'll do the same approach as baseclass: compute final metrics
        return super()._compute_metrics(y_test, predictions, self.predict_proba(X_test))

    def predict(self, X_test):
        """
        Return predicted class via majority vote among the approximate neighbors.
        """
        preds = []
        for x in X_test:
            # Retrieve approximate neighbors
            nn_indices = self.annoy_index.get_nns_by_vector(x, self.n_neighbors)
            # majority voting among neighbors
            neighbor_labels = self.y_train[nn_indices]
            # pick the most common label
            vals, counts = np.unique(neighbor_labels, return_counts=True)
            pred = vals[np.argmax(counts)]
            preds.append(pred)
        return np.array(preds)

    def predict_proba(self, X_test):
        """
        Approximate probability: among the neighbors, how many are class=1?
        This only works if we have a binary classification 0/1. If multi-class, you'd need a more advanced approach.
        """
        # For multi-class, you'd sum up each class count. But here's a binary example:
        probs = []
        for x in X_test:
            nn_indices = self.annoy_index.get_nns_by_vector(x, self.n_neighbors)
            neighbor_labels = self.y_train[nn_indices]
            # For binary classification (0 or 1)
            p = np.mean(neighbor_labels)  # fraction of 1's in neighbors
            probs.append([1-p, p])  # shape => [prob_of_0, prob_of_1]
        return np.array(probs)

    def _compute_metrics(self, y_true, y_pred, y_proba):
        """
        If you want to override the base metrics, you can do it here.
        But by default we can rely on the parent's evaluate logic.
        """
        pass
