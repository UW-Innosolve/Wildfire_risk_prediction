from feat_engineer import FeatEngineer
from preprocessor import Preprocessor

class FbDataset(FeatEngineer, Preprocessor):
  def __init__(self, raw_data_df):
    self.data = raw_data_df
    
  def gen_features(self):
    pass
    
  def process(self):
    pass