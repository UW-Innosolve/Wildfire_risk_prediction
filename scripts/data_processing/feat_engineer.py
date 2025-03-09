import pandas as pd
from preprocessor import Preprocessor
from ..feature_engineering.temporal import FbTemporalFeatures

class FeatEngineer(FbTemporalFeatures):
  def __init__(self, raw_data_df):
    self.raw_data = raw_data_df
    self.data_features = pd.DataFrame()
    
  def apply(self, temporal=True):
    if temporal:
      temporal_feats = FbTemporalFeatures.features
      self.data_features = self.data_features + temporal_feats 
      ## NOTE: Ensure this appending of the new feature cols is corrects
    
    