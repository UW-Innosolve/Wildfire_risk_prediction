from sklearn.linear_model import LinearRegression, LogisticRegression

## Inluding linear, polynomial, and logisitc regression modeling.
## consider other models.

class FbRegressionModel(BaseModel):
  def __init__(self, input_df):
    self.x_data = input_df
    
    ## init hyperparams for lin:
    
    ## init hyperparams for poly:

    ## init hyperparams for log:


  def config_lin_reg(self):
    pass
  
  def config_poly_reg(self):
    pass
  
  def config_log_reg(self):
    pass
  
  # NOTE: All model types should have default setups.
  def train(self, x_train, y_train,
            lin_reg=True, poly_reg=True, log_reg=True):
    pass