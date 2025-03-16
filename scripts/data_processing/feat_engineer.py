import pandas as pd
import logging
from preprocessor import Preprocessor
from feature_engineering.temporal import FbTemporalFeatures
from feature_engineering.spatial import FbSpatialFeatures
from feature_engineering.weather import FbWeatherFeatures
from feature_engineering.surface import FbSurfaceFeatures
from feature_engineering.cwfdrs import FbCwfdrsFeatures

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatEngineer(FbTemporalFeatures, FbSpatialFeatures, FbWeatherFeatures, FbSurfaceFeatures, FbCwfdrsFeatures):
  def __init__(self, raw_data_df):
    self.raw_data = raw_data_df.copy()
    self.data_features = self.raw_data[['date', 'latitude', 'longitude']].copy() # Start with the date, latitude, longitude index
    
  def apply_features(self, eng_feats):
    logger.info(f"Applying engineered features: {eng_feats}")
    if eng_feats == ['DISABLE']:
      return
    
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
    # NOTE: The weather feature set is computed by the FbWeatherFeatures class, stored in the weather_features member until get_features() is called.
    if any([feat in eng_feats for feat in ['lightning_products', 'lightning_ratios', 'rolling_precipitation', 'relative_humidity', 'atmospheric_dryness']]):
      self.weather = FbWeatherFeatures(self.raw_data)
      if 'lightning_products' in eng_feats:
        self.weather.lightning_products()
      if 'lightning_ratios' in eng_feats:
        self.weather.lightning_to_ratios()
      if 'rolling_precipitation' in eng_feats:
        self.weather.rolling_precipitation()
      if 'relative_humidity' in eng_feats:
        self.weather.rel_humidity()
      if 'atmospheric_dryness' in eng_feats:
        self.weather.rel_atmospheric_dryness()
        
      self.weather_df  = self.weather.get_features()
      # for col in self.weather_df.columns:
      #   print(f"col: {col}")
      #   print(self.weather_df[col])
      
      self.data_features = pd.merge(self.data_features, self.weather_df,
                                    on=['date', 'latitude', 'longitude'], how='outer')
      
    # Surface features
    if any([feat in eng_feats for feat in ['fuel_low', 'fuel_high', 'soil', 'elevation', 'x_slope', 'y_slope', 'surface_depth_waterheat', 'surface_water_sum', 'surface_heat_sum']]):
      self.surface = FbSurfaceFeatures(self.raw_data)
      if 'fuel_low' in eng_feats or 'fuel_high' in eng_feats:
        self.surface.vegetation()
      if 'soil' in eng_feats:
        self.data_features['soil'] = self.surface.soil()
      if 'surface_depth_waterheat' in eng_feats or 'surface_water_sum' in eng_feats or 'surface_heat_sum' in eng_feats:
        self.surface.surface_depth_waterheat()
      if 'elevation' in eng_feats or 'x_slope' in eng_feats or 'y_slope' in eng_feats:
        self.surface.topography()
        
      self.data_features = self.data_features + self.surface.get_features()
    
    # Temporal features
    if any([feat in eng_feats for feat in ['season', 'fire_season']]):
      self.temporal = FbTemporalFeatures(self.raw_data)
      self.temporal_df = self.data_features[['date', 'latitude', 'longitude']].copy()
      
      if 'season' in eng_feats:
        self.temporal_df = pd.merge(self.temporal_df, self.temporal.seasonal(),
                                              on=['date', 'latitude', 'longitude'], how='outer')
        logger.info(f"temporal_df shape: {self.temporal_df.shape}")
      if 'fire_season' in eng_feats:
        self.temporal_df = pd.merge(self.temporal_df, self.temporal.fire_seasonal(),
                                                   on=['date', 'latitude', 'longitude'], how='outer')
        logger.info(f"temporal_df shape: {self.temporal_df.shape}")
      
      self.data_features = pd.merge(self.data_features, self.temporal_df,
                                    on=['date', 'latitude', 'longitude'], how='outer')
      
    # Spatial features
    if any([feat in eng_feats for feat in ['clusters_12', 'clusters_30']]):
      self.spatial = FbSpatialFeatures(self.raw_data)
      self.spatial_df = self.data_features[['date', 'latitude', 'longitude']].copy()
      if 'clusters_12' in eng_feats:
        self.spatial_df['clusters_12'] = self.spatial.kmeans_cluster(n_clusters=12)
        logger.info(f"spatial_df shape: {self.spatial_df.shape}") 
      if 'clusters_36' in eng_feats:
        self.spatial_df['clusters_30'] = self.spatial.kmeans_cluster(n_clusters=36)
        logger.info(f"spatial_df shape: {self.spatial_df.shape}")
        
      self.data_features = pd.merge(self.data_features, self.spatial_df, on=['date', 'latitude', 'longitude'], how='outer')
      
    print(self.data_features.head())
      
    return self.data_features
    
    
    