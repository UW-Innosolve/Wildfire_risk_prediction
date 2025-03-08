from sklearn.ensemble import RandomForestClassifier
from model_evaluation.model_metrics import BaseModel

class RandomForestModel(BaseModel):
    def __init__(self, params=None):
        """
        Initialize Random Forest Model.
        Default parameters:
          - n_estimators=100: Number of trees for the ensemble.
          - max_depth=None: Allow trees to grow fully (can be tuned to prevent overfitting).
          - class_weight='balanced': Helps adjust for imbalanced classes.
          - random_state=42: Ensures reproducibility.
        """
        if params is None:
            params = {'n_estimators': 100, 'max_depth': None, 'class_weight': 'balanced', 'random_state': 42}
        self.model = RandomForestClassifier(**params)

    def train(self, X_train, y_train):
        """
        Train the Random Forest model.
        """
        self.model.fit(X_train, y_train)
