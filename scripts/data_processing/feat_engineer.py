import pandas as pd
from preprocessor import Preprocessor
from ..feature_engineering.temporal import FbTemporalFeatures
from ..feature_engineering.spatial import FbSpatialFeatures
from ..feature_engineering.weather import FbWeatherFeatures
from ..feature_engineering.surface import FbSurfaceFeatures

class FeatEngineer(FbTemporalFeatures, FbSpatialFeatures, FbWeatherFeatures, FbSurfaceFeatures):
  def __init__(self, raw_data_df):
    self.raw_data = raw_data_df
    self.data_features = pd.DataFrame()
    
  def apply_features(self, 
            temporal=True, spatial=True,
            weather=True, surface=True):
    if temporal:
      temporal_feats = FbTemporalFeatures.features
      self.data_features = self.data_features + temporal_feats
      
    if spatial:
      spatial_feats = FbSpatialFeatures.features
      self.data_features = self.data_features + spatial_feats
      
    if weather:
      weather_feats = FbWeatherFeatures.features
      self.data_features = self.data_features + weather_feats
      
    if surface:
      surface_feats = FbSurfaceFeatures.features
      self.data_features = self.data_features + surface_feats
    
    
    
    