from model_evaluation.model_metrics import BaseModel
import xgboost as xgb


class XGBoostModel(BaseModel):
    def __init__(self, params=None):
        """
        Initialize XGBoost Model.
        Default parameters:
          - learning_rate=0.1: Standard learning rate for convergence.
          - max_depth=6: Controls complexity (tune if overfitting is observed).
          - n_estimators=100: Number of boosting rounds.
          - subsample=0.8: Uses 80% of data per iteration to reduce overfitting.
          - scale_pos_weight=5: Empirically set to handle class imbalance.
          - use_label_encoder=False and eval_metric='logloss': Avoids warnings and sets an appropriate metric.
        """
        if params is None:
            params = {
                'learning_rate': 0.1,
                'max_depth': 6,
                'n_estimators': 100,
                'subsample': 0.8,
                'scale_pos_weight': 5,
                'use_label_encoder': False,
                'eval_metric': 'logloss'
            }
        self.model = xgb.XGBClassifier(**params)

    def train(self, X_train, y_train):
        """
        Train the XGBoost model.
        """
        self.model.fit(X_train, y_train)
