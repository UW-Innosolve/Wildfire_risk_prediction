from sklearn.linear_model import LogisticRegression
from model_evaluation.model_metrics import BaseModel

class LogisticRegressionModel(BaseModel):
    def __init__(self, params=None):
        """
        Initialize Logistic Regression Model.
        Default parameters:
          - C=1.0: Regularization strength (lower values = stronger regularization).
          - solver='liblinear': Suitable for smaller datasets.
          - class_weight='balanced': Adjusts weights inversely proportional to class frequencies.
          - max_iter=1000: Increases iterations to ensure convergence.
        """
        if params is None:
            params = {'C': 1.0, 'solver': 'liblinear', 'class_weight': 'balanced', 'max_iter': 1000}
        self.model = LogisticRegression(**params)

    def train(self, X_train, y_train):
        """
        Train the logistic regression model.
        """
        self.model.fit(X_train, y_train)
