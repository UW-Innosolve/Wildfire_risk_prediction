import pandas as pd
from preprocessor import Preprocessor
from ..feature_engineering.temporal import FbTemporalFeatures
from ..feature_engineering.spatial import FbSpatialFeatures
from ..feature_engineering.weather import FbWeatherFeatures
from ..feature_engineering.surface import FbSurfaceFeatures
from ..feature_engineering.cwfdrs import FbCwfdrsFeatures

class FeatEngineer(FbTemporalFeatures, FbSpatialFeatures, FbWeatherFeatures, FbSurfaceFeatures, FbCwfdrsFeatures):
  def __init__(self, raw_data_df):
    self.raw_data = raw_data_df
    self.data_features = pd.DataFrame()
    
  def apply_features(self, eng_feats):
    
    # Fire weather danger rating system features
    if any([feat in eng_feats for feat in ['drought_code', 'duff_moisture_code', 'fine_fuel_moisture_code', 'initial_spread_index', 'build_up_index', 'fire_weather_index']]):
      self.cwfdrs = FbCwfdrsFeatures(self.raw_data)
      if 'drought_code' in eng_feats:
        self.data_features['drought_code'] = self.cwfdrs['drought_code']
      if 'duff_moisture_code' in eng_feats:
        self.data_features['duff_moisture_code'] = self.cwfdrs['duff_moisture_code']
      if 'fine_fuel_moisture_code' in eng_feats:
        self.data_features['fine_fuel_moisture_code'] = self.cwfdrs['fine_fuel_moisture_code']
      if 'initial_spread_index' in eng_feats:
        self.data_features['initial_spread_index'] = self.cwfdrs['initial_spread_index']
      if 'build_up_index' in eng_feats:
        self.data_features['build_up_index'] = self.cwfdrs['build_up_index']
      if 'fire_weather_index' in eng_feats:
        self.data_features['fire_weather_index'] = self.cwfdrs['fire_weather_index']

    # Weather features
    if any([feat in eng_feats for feat in ['lightning_products', 'lightning_ratios', 'rolling_precipitation', 'relative_humidity', 'atmospheric_dryness']]):
      self.weather = FbWeatherFeatures(self.raw_data)
      if 'lightning_products' in eng_feats:
        self.data_features['lightning_products'] = self.weather.lightning_products(self.raw_data)
      if 'lightning_ratios' in eng_feats:
        self.data_features['lightning_ratios'] = self.weather.lightning_ratios(self.raw_data)
      if 'rolling_precipitation' in eng_feats:
        self.data_features['rolling_precipitation'] = self.weather.rolling_precipitation(self.raw_data)
      if 'relative_humidity' in eng_feats:
        self.data_features['relative_humidity'] = self.weather.relative_humidity(self.raw_data)
      if 'atmospheric_dryness' in eng_feats:
        self.data_features['atmospheric_dryness'] = self.weather.atmospheric_dryness(self.raw_data)
      
    # Surface features
    if any([feat in eng_feats for feat in ['fuel_low', 'fuel_high', 'soil', 'surface_depth_waterheat', 'elevation', 'x_slope', 'y_slope']]):
      self.surface = FbSurfaceFeatures(self.raw_data)
      if 'fuel_low' in eng_feats:
        self.data_features['fuel_low'] = self.surface.fuel_low(self.raw_data)
      if 'fuel_high' in eng_feats:
        self.data_features['fuel_high'] = self.surface.fuel_high(self.raw_data)
      if 'soil' in eng_feats:
        self.data_features['soil'] = self.surface.soil(self.raw_data)
      if 'surface_depth_waterheat' in eng_feats:
        self.data_features['surface_depth_waterheat'] = self.surface.surface_depth_waterheat(self.raw_data)
      if 'elevation' in eng_feats:
        self.data_features['elevation'] = self.surface.elevation(self.raw_data)
      if 'x_slope' in eng_feats:
        self.data_features['x_slope'] = self.surface.x_slope(self.raw_data)
      if 'y_slope' in eng_feats:
        self.data_features['y_slope'] = self.surface.y_slope(self.raw_data)
    
    # Temporal features
    if any([feat in eng_feats for feat in ['season', 'fire_season']]):
      self.temporal = FbTemporalFeatures(self.raw_data)
      if 'season' in eng_feats:
        self.data_features['season'] = self.temporal.season(self.raw_data)
      if 'fire_season' in eng_feats:
        self.data_features['fire_season'] = self.temporal.fire_season(self.raw_data)
      
    # Spatial features
    if any([feat in eng_feats for feat in ['clusters_12', 'clusters_24', 'clusters_36']]):
      self.spatial = FbSpatialFeatures(self.raw_data)
      if 'clusters_12' in eng_feats:
        self.data_features['clusters_12'] = self.spatial.kmeans_cluster(n_clusters=12)
      if 'clusters_24' in eng_feats:
        self.data_features['clusters_24'] = self.spatial.kmeans_cluster(n_clusters=24)
      if 'clusters_36' in eng_feats:
        self.data_features['clusters_36'] = self.spatial.kmeans_cluster(n_clusters=36)
    
    
    
    