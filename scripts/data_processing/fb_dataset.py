from feat_engineer import FeatEngineer
from preprocessor import Preprocessor
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FbDataset(FeatEngineer, Preprocessor):
  def __init__(self):
    self.fb_model_features = pd.DataFrame() # Includes all features (raw and engineered)
    self.fb_processed_data = pd.DataFrame() # Includes all features (raw and engineered) after processing
    
  # NOTE: THESE ARE FEATURE DEFAULTS, DO NOT EDIT THESE.
  ## Instead, override the parameters using the config_features method
  ## to create an instance with a custom features configuration.
  
  # config_features
  #   raw_params_feats: list of raw parameters to be included as features
  #   eng_feats: list of engineered features to be included
  #   Does not return anything, must be called before generate_features
  def config_features(self, raw_params_feats=None, eng_feats=None):
    self.raw_param_feats = ['latitute', 'longitude', 
                            '10u',	'10v', '2d', '2t',  # wind, dewpoint, temperature
                            'cl', # Lake cover              
                            'cvh', 'cvl', # Low vegetation cover, high vegetation cover
                            'fal', # Forecast albedo
                            'lai_hv',	'lai_lv', # Leaf area index high vegetation, low vegetation
                            'lsm', # Land-sea mask
                            # 'slt', # Soil type - NOTE: soil feature is preferred.
                            'sp',	# Surface pressure
                            'src', # Skin reservoir content
                            # NOTE: stl and swv layers are excluded in favour of waterheat ratio, daily_water_sum,  daily_temp_sum
                            'stl1',	'stl2',	'stl3',	'stl4', # Soil temperature levels (0-7cm, 7-28cm, 28-100cm, 100-289cm)
                            # 'swvl1',	'swvl2',	'swvl3',	'swvl4', # Soil water volume levels (0-7cm, 7-28cm, 28-100cm, 100-289cm)
                            # 'tvh',	'tvl', # High vegetation type, low vegetation type (Categorical) #NOTE: fuel_low and fuel_high are preferred
                            # 'z', # Geopotential (proportional to elevation) #NOTE: elevation is preferred
                            'e',	'pev', # Evaporation, potential evaporation
                            'slhf',	'sshf',# Latent heat flux, sensible heat flux
                            'ssr',	'ssrd', # surface solar radiation, surface solar radiation down
                            'str',	'strd',	# Surface thermal radiation, surface thermal radiation down
                            'tp', # Total precipitation
                            'sf', # Snowfall
                            'is_fire_day', # NOTE: This is the target variable
                            'lightning_count', 'absv_strength_sum',	'multiplicity_sum', # Lightning count, absolute strength sum, multiplicity sum
                            'railway_count',	'power_line_count',	'highway_count', # Railway, power line, highway count
                            'aeroway_count',	'waterway_count' # Aeroway, waterway count
                          ]
    
    self.eng_feats =            [ # CWFDRS Fire weather indices
                                  'drought_code', 'duff_moisture_code',
                                  'fine_fuel_moisture_code', 'initial_spread_index',
                                  'build_up_index', 'fire_weather_index',
                                  
                                  # Weather features
                                  #  - Lightning
                                  'lightning_products', 'lightning_ratios',
                                  #  - Precipitation
                                  'rolling_precipitation',
                                  #  - Atmospheric
                                  'relative_humidity', 'atmospheric_dryness',
                                  
                                  # Surface features
                                  #  - Fuel (From vegetation)
                                  'fuel_low', 'fuel_high',
                                  #  - Soil
                                  'soil', # Catagorical
                                  #  - Surface water and heat
                                  'surface_depth_waterheat', # Adds 17 columns, with ratio for each 17cm depth
                                  #  - Topography
                                  'elevation',
                                  'x_slope',
                                  'y_slope'
                                  
                                  # Temporal features
                                  # - Seasonal
                                  'season', 'fire_season',
                                  
                                  # Spatial features
                                  'clusters_12', 'clusters_24', 'clusters_36'
                                ]
    if raw_params_feats:
      self.raw_param_feats = raw_params_feats
    if eng_feats:
      self.eng_feats = eng_feats
      
  # Generate features
  #   - Generates features from raw data and engineered features (as set in config_features)
  #   - Returns a DataFrame with all features to be used in the model
  def generate_features(self):
    self.fb_model_features = self.raw_data[self.raw_param_feats]
    
    self.feat_engineer = FeatEngineer(self.raw_data)
    self.fb_model_features = self.fb_model_features + self.feat_engineer.apply_features(self.eng_feats)
    
    return self.fb_model_features
    
  # Process
  #   - Processes the raw data and generates features (both raw and engineered)
  #   - Scales and one-hot encodes features, minmax, standard, and onehot set intentionally
  #   - Inputs: data_dir (str), raw_params_feats (list), eng_feats (list) - if not set, defaults are used
  #   - Returns: processed data (DataFrame)
  def process(self, data_dir, raw_params_feats=None, eng_feats=None):
    # Initialize the Preprocessor.
    self.preprocessor = Preprocessor(data_dir)
    logger.info("Loading data from CSV files...")
    self.raw_data = self.preprocessor.load_data()  # Aggregate CSVs.
    
    logger.info("Cleaning data (converting dates, removing missing target values)...")
    self.raw_data = self.preprocessor.clean_data()  # Clean the data (mutates the data member in preprocessor instance)
    
    ## Feature Engineering (relies on raw_data member set above)
    self.config_features(raw_params_feats, eng_feats) # Set the features to be used (or use defaults if not set)
    self.fb_model_features_raw = self.generate_features()
    
    
    # Define feature list for scaling type and onehotting.
    self.numeric_features_ss = [
                                '2t', '2d', '10u', '10v', 'sp', # wind, dewpoint, temperature, surface pressure
                                'tp', 'rolling_precipitation', # Total precipitation, rolling precipitation
                                'e', 'pev', # Evaporation, potential evaporation
                                'slhf', 'sshf', # Latent heat flux, sensible heat flux
                                'ssr', 'ssrd', # surface solar radiation, surface solar radiation down
                                'str', 'strd', # Surface thermal radiation, surface thermal radiation down
                                'tp', 'sf', # Total precipitation, snowfall
                                'lightning_count', 'absv_strength_sum', 'multiplicity_sum', # Lightning count, absolute strength sum, multiplicity sum
                                'stl1', 'stl2', 'stl3', 'stl4', # Soil temperature levels (0-7cm, 7-28cm, 28-100cm, 100-289cm)
                                # Fire weather indices unbounded
                                'drought_code', 'duff_moisture_code', 'fine_fuel_moisture_code', 'initial_spread_index', 'build_up_index', 'fire_weather_index',
                                'ltng_multiplicity_prod', 'ltng_strength_prod', # Lightning products
                                'relative_humidity', 'atmospheric_dryness', # Relative humidity, atmospheric dryness
                                'rolling_precipitation', # Rolling precipitation
                                'daily_water_sum', 'daily_temp_sum', # Daily water and temperature sum from the surface
                                ]
    
    self.numeric_features_mm = [ # MinMax scaling
                                 # Best for proportions, ratios, and non-Gaussian distributions
                                'cl', # Lake cover
                                'cvh', 'cvl', # Low vegetation cover, high vegetation cover
                                'fal', # Forecast albedo
                                'lai_hv', 'lai_lv', # Leaf area index high vegetation, low vegetation
                                'lsm', # Land-sea mask
                                'src', # Skin reservoir content                                                                                                                                                                                                
                                'railway_count', 'power_line_count', 'highway_count', 'aeroway_count', 'waterway_count', # Infrastructure not normally distributed
                                'ltng_multiplicity_ratio', 'ltng_strength_ratio', # Lightning ratios
                                'surface_depth_waterheat', # Waterheat ratio
                                'elevation', # Elevation
                                'x_slope', 'y_slope', # Slope
                                
                                'fuel_low', 'fuel_high', # NOTE: Fuel type is binary (remains unchanged duing minmax scaling)
                                'is_fire_day' # Target variable is binary (remains unchanged during minmax scaling)
                                ]
    
    self.catgorical_features = [
                                'soil', # Soil type
                                'season', 'fire_season', # Seasonal features
                                'clusters_12', 'clusters_24', 'clusters_36' # Spatial clustering
                                ]
    
    for num_feat in self.numeric_features_ss:
      self.fb_processed_data = self.fb_processed_data + self.preprocessor.scale_features_ss([num_feat])
    
    for num_feat in self.numeric_features_mm:
      self.fb_processed_data = self.fb_processed_data + self.preprocessor.scale_features_minmax([num_feat])
      
    for cat_feat in self.catgorical_features:
      self.fb_processed_data = self.fb_processed_data + self.preprocessor.onehot_cat_features([cat_feat])
    
    return self.fb_processed_data
  
  
  # Split
  #   - Splits the processed data into training and test sets
  #   - Applies SMOTE for balancing the minority class (fire days)
  #   - Inputs: test_size (float), random_state (int), apply_smote (bool)
  #   - Returns: X_train, X_test, y_train, y_test (DataFrames)
  def split(self, test_size=0.2, random_state=42, apply_smote=True):
    # Define the target variable
    target_param = 'is_fire_day'
    target = self.fb_processed_data[target_param]
    
    # Get the feature list
    feature_list = self.fb_processed_data.columns.tolist()
    
    # Split the data; apply SMOTE for balancing minority class (fire days).
    logger.info("Splitting data into training and test sets and applying SMOTE for balancing...")
    X_train, X_test, y_train, y_test = self.preprocessor.split_data(feature_list, target, apply_smote=apply_smote)
    logger.info(f"Training set size: {X_train.shape}, Test set size: {X_test.shape}")
    
    self.X_train = X_train
    self.X_test = X_test
    self.y_train = y_train
    self.y_test = y_test
    
    return X_train, X_test, y_train, y_test
  
  def get_processed_data(self):
    return self.fb_processed_data
  
  def get_split_data(self):
    return self.X_train, self.X_test, self.y_train, self.y_test
  
  

      
    
    
    