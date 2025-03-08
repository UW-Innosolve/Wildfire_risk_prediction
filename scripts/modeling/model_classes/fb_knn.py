from model_evaluation.model_metrics import BaseModel
from sklearn.neighbors import KNeighborsClassifier


class KNNModel(BaseModel):
    def __init__(self, n_neighbors=5, weights='uniform'):
        """
        Initialize KNN Model.
        Parameters:
          - n_neighbors: Number of neighbors (default=5; can be tuned, e.g., 10, 15).
          - weights: 'uniform' gives equal weight; 'distance' weights by proximity.
        """
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)

    def train(self, X_train, y_train):
        """
        Train the KNN model.
        """
        self.model.fit(X_train, y_train)
