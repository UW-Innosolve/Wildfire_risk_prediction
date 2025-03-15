from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV
from sklearn.preprocessing import PolynomialFeatures
from model_evaluation.model_metrics import BaseModel
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LinearRegressionModel(BaseModel):
    def __init__(self, params=None):
        """
        Initialize Linear Regression Model.
        Default parameters: None.
        """
        if params is None:
            params = {}
        self.lin_reg_model = LinearRegression(**params)

    def train(self, X_train, y_train):
        """
        Train the linear regression model.
        """
        self.lin_reg_model.fit(X_train, y_train)
        
        
class PolynomialRegressionModel(BaseModel):
    def __init__(self, params=None):
        """
        Initialize Polynomial Regression Model.
        Default parameters: None.
        """
        if params is None:
            params = {'degree': 3}
        self.poly_reg_model = PolynomialFeatures(**params)
        
    def train(self, X_train, y_train):
        """
        Train the polynomial regression model.
        """
        self.poly_reg_model.fit(X_train, y_train)


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
        self.log_reg_model.fit(X_train, y_train)
        

class RidgeRegressionModel(BaseModel):
    def __init__(self, params=None, alphas=None):
        """
        Initialize Ridge Regression Model.
        Default parameters:
          - solver='auto': Automatically selects the solver based on the data.
          - tol=1e-4: Tolerance for stopping criteria.
        """
        
        if alphas is None:
            alphas = np.array([1.e-06, 1.e-05, 1.e-04, 1.e-03, 1.e-02, 1.e-01, 1.e+00, 1.e+01,
                            1.e+02, 1.e+03, 1.e+04, 1.e+05, 1.e+06])
        if params is None:
            params = {'solver': 'auto', 'tol': 1e-4, 'alphas': alphas}
        self.ridge_reg_model = RidgeCV(**params)
        
    def train(self, X_train, y_train):
        """
        Train the ridge regression model.
        """
        self.ridge_reg_model.fit(X_train, y_train)
        
  
  
  
  
# ## Regression model class
# ## Includes Linear, Polynomial, Logistic, and Ridge regression models
# class FbRegressionModel(BaseModel):
#   def __init__(self):
#     logger.info("Initializing FbRegressionModel")

#   def config_lin_reg(self):
#     self.lin_reg_model = LinearRegression(**self.lin_reg_params)
  
#   def config_poly_reg(self, degree=3):
#     self.poly_reg_params = {'degree': degree}
#     self.poly_reg_model = PolynomialFeatures(**self.poly_reg_params)
    
    
#   def config_log_reg(self, c=1.0, solver='liblinear', class_weight='balanced', max_iter=1000, tol=1e-4):
#     self.log_reg_params = {'C': c, 'solver': solver, 'class_weight': class_weight, 'max_iter': max_iter, 'tol': tol}
#     self.log_reg_model = LogisticRegression(**self.log_reg_params)
    
#   def config_ridge_reg(self, solver='auto', tol=1e-4):
#     alphas= np.array([1.e-06, 1.e-05, 1.e-04, 1.e-03, 1.e-02, 1.e-01, 1.e+00, 1.e+01,
#       1.e+02, 1.e+03, 1.e+04, 1.e+05, 1.e+06])
#     self.ridge_reg_params = {'alphas': alphas, 'solver': solver, 'tol': tol}
#     self.ridge_reg_model = RidgeCV(**self.ridge_reg_params) 
#     # RidgeCV automatically performs cross-validation (on the training data) to find the optimal alpha value.
  
#   # NOTE: All model types should have default setups.
#   def train(self, x_train, y_train,
#             lin_reg=True, poly_reg=True, log_reg=True, ridge_reg=True):
#     if lin_reg:
#       self.lin_reg_model.fit(x_train, y_train)
#     if poly_reg:
#       self.poly_reg_model.fit(x_train, y_train)
#     if log_reg:
#       self.log_reg_model.fit(x_train, y_train)
#     if ridge_reg:
#       self.ridge_reg_model.fit(x_train, y_train)