from feat_engineer import FeatEngineer
from preprocessor import Preprocessor
from sklearn.model_selection import train_test_split
import pandas as pd
import logging
import os
import glob


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FbDataset(FeatEngineer, Preprocessor):
  def __init__(self, raw_data_dir):
    logger.info(f"Initializing FbDataset with data directory: {raw_data_dir}")
    # Initialize the raw data directory and load the raw data.
    logger.info("Loading data from CSV files...")
    self.raw_data_dir = raw_data_dir
    self.raw_data = self._load_data(data_dir=self.raw_data_dir)
    # Ensure the 'date' column is of type datetime64[ns] in both DataFrames
    self.raw_data['date'] = pd.to_datetime(self.raw_data['date'])
    
    self.fb_model_features = self.raw_data[['date', 'latitude', 'longitude']].copy()  # Includes all features (raw and engineered)
    self.fb_model_features_raw = pd.DataFrame() # Includes all features (from-raw and engineered) before processing
    self.fb_processed_data = pd.DataFrame() # Includes all features (from-raw and engineered) after processing

    # Initialize the raw data directory and load the raw data.
    logger.info("Loading data from CSV files...")
    self.raw_data_dir = raw_data_dir
    self.raw_data = self._load_data(data_dir=self.raw_data_dir)
    # Ensure the 'date' column is of type datetime64[ns] in both DataFrames
    self.raw_data['date'] = pd.to_datetime(self.raw_data['date'])

  def _load_data(self, data_dir):
    """
    Aggregate all CSV files from the specified directory into a single DataFrame.
    Uses glob to fetch file paths and concatenates them.
    """
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    logger.info((f"Found {len(csv_files)} CSV files in directory: {self.raw_data_dir}"))
    dfs = []
    for file in csv_files:
        try:
            df_temp = pd.read_csv(file)
            logger.debug(f"Loaded {os.path.basename(file)} with shape {df_temp.shape} onto raw df")
            dfs.append(df_temp)
        except Exception as e:
            logger.error(f"Error loading {file}: {e}")
    if dfs:
        data = pd.concat(dfs, ignore_index=True)
        logger.info("Aggregated DataFrame shape:", data.shape)
    else:
        raise ValueError("No CSV files found in the specified directory.")
      
    return data
    
  
  # NOTE: BELOW ARE FEATURE DEFAULTS, DO NOT EDIT THESE.
  ## Instead, override the parameters using the config_features method
  ## to create an instance with a custom features configuration.
  
  # config_features
  #   raw_params_feats: list of raw parameters to be included as features
  #   eng_feats: list of engineered features to be included
  #   Does not return, must be called before generate_features
  def config_features(self, raw_params_feats=None, eng_feats=None):
    self.raw_param_feats = [
                            # NOTE Spatial and temporal index included here but not in the final processed data.
                            # 'latitude', 'longitude', 
                            # 'date', # Date
                            # -----------------------------------------------------------------------------------
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
                            # 'sf', # Snowfall NOTE: Excluded for now, can be used in the definition of fire season (CWFDRS)
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
                                  'clusters_12', 'clusters_30'
                                ]
    if raw_params_feats or (raw_params_feats == ['DISABLE']): # Passing 'DISABLE' will cause no raw parameters to be used
      self.raw_param_feats = raw_params_feats
    if eng_feats or (eng_feats == ['DISABLE']): # Passing 'DISABLE' will cause no feature engineering to be used
      self.eng_feats = eng_feats
      
      
  # Generate features
  #   - Generates features from raw data and engineered features (as set in config_features)
  #   - Returns a DataFrame with all features to be used in the model
  def generate_features(self):
    logger.info(f"Using raw data features{self.raw_param_feats}")
    logger.info(f"Using engineered features: {self.eng_feats}")
    
    if self.raw_param_feats == ['DISABLE']:
      logging.info("No raw parameters selected as features, using only engineered features.")
    else:
      features_raw = self.raw_param_feats + ['date', 'latitude', 'longitude']
      self.fb_model_features = pd.merge(self.fb_model_features, self.raw_data[features_raw],
                                        on=['date', 'latitude', 'longitude'], how='outer')
    
    self.feat_engineer = FeatEngineer(self.raw_data)
    self.engineered_feats = self.feat_engineer.apply_features(self.eng_feats)
    
    if not self.engineered_feats.empty: # Since apply_features returns None if 'DISABLE' is passed
      self.fb_model_features = pd.merge(self.fb_model_features,
                                        self.engineered_feats,
                                        on=['date', 'latitude', 'longitude'], how='outer')
    
    return self.fb_model_features
    
    
  # Process
  #   - Processes the raw data and generates features (both raw and engineered)
  #   - Scales and one-hot encodes features, minmax, standard, and onehot set intentionally
  #   - Inputs: data_dir (str), raw_params_feats (list), eng_feats (list) - if not set, defaults are used
  #   - Returns: processed data (DataFrame)
  def process(self):
    # Initialize the Preprocessor.
    self.preprocessor = Preprocessor(raw_data_df=self.raw_data)

    logger.info("Cleaning data (converting dates, removing missing target values)...")
    self.raw_data = self.preprocessor.clean_data()  # Clean the data (mutates the data member in preprocessor instance, and returns it)
    
    ## Feature Engineering (relies on raw_data member set at initialization)
    self.config_features(self.raw_param_feats, self.eng_feats) # Set the features to be used (or use defaults if not set)
    self.fb_model_features_raw = self.generate_features()
    
    # Define feature list for scaling type and onehotting.
    self.numeric_features_ss = [
                                '2t', '2d', '10u', '10v', 'sp', # wind, dewpoint, temperature, surface pressure
                                'tp', # Total precipitation, rolling precipitation
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
    
    self.categorical_features = [
                                'soil', # Soil type
                                'season', 'fire_season', # Seasonal features
                                'clusters_12', 'clusters_30' # Spatial clustering
                                ]
    
    # Filter the feature lists to only include features present in fb_model_features_raw.columns
    self.numeric_features_ss = [feature for feature in self.numeric_features_ss if feature in self.fb_model_features_raw.columns]
    self.numeric_features_mm = [feature for feature in self.numeric_features_mm if feature in self.fb_model_features_raw.columns]
    self.categorical_features = [feature for feature in self.categorical_features if feature in self.fb_model_features_raw.columns]
    
    # Initialize the processed data with the index: 'date', 'latitude', 'longitude'.
    self.fb_processed_data = self.raw_data[['date', 'latitude', 'longitude']]
    
    # Scale and one-hot encode features
    fb_model_feat_raw_ss = self.fb_model_features_raw[self.numeric_features_ss]
    fb_model_feat_processed_ss = self.preprocessor.scale_features_ss(fb_model_feat_raw_ss)
    self.fb_processed_data = pd.merge(self.fb_processed_data, fb_model_feat_processed_ss, 
                                      on=['date', 'latitude', 'longitude'], how='outer')
    logger.info(f"Data shape after aggregation of standard scaling numeric features: {self.fb_processed_data.shape}")
      
    fb_model_feat_raw_mm = self.fb_model_features_raw[self.numeric_features_mm]
    fb_model_feat_processed_mm = self.preprocessor.scale_features_mm(fb_model_feat_raw_mm)
    self.fb_processed_data = pd.merge(self.fb_processed_data, fb_model_feat_processed_mm,
                                      on=['date', 'latitude', 'longitude'], how='outer')
    logger.info(f"Data shape after aggregation of MinMax scaling numeric features: {self.fb_processed_data.shape}")
    
    print(self.categorical_features)
    fb_model_feat_raw_onehot = self.fb_model_features_raw[self.categorical_features]
    fb_model_feat_processed_onehot = self.preprocessor.onehot_cat_features(fb_model_feat_raw_onehot)
    self.fb_processed_data = pd.merge(self.fb_processed_data, fb_model_feat_processed_onehot,
                                      on=['date', 'latitude', 'longitude'], how='outer')
    logger.info(f"Data shape after aggregation of one-hot encoded features: {self.fb_processed_data.shape}")
    
    logger.info("Data processing complete.")
    logger.info(f"Final processed data shape: {self.fb_processed_data.shape}")
    
    return self.fb_processed_data
  
  
  # Split
  #   - Splits the processed data into training and test sets
  #   - Applies SMOTE for balancing the minority class (fire days)
  #   - Inputs: test_size (float), random_state (int), apply_smote (bool)
  #   - Returns: X_train, X_test, y_train, y_test (DataFrames) NOTE: (OUTPUT DATA DOES NOT HAVE INDEX COLUMNS 'date', 'latitude', 'longitude')
  def split(self, test_size=0.2, random_state=42, apply_smote=True):
    """
    Split the data into training and testing sets.
      - Optionally apply SMOTE to the training data.
      - Parameters like test_size and random_state can be adjusted.
    """
    
    data = self.fb_processed_data.copy()
    
    # Define the target variable
    target_param = 'is_fire_day'
    
    # Get the feature list
    feature_list = data.columns.tolist()
    try:
        feature_list.remove('is_fire_day')
        logger.debug("Removed target variable from feature list.")
    except ValueError:
        logger.error("Target variable not found in feature list, or could not be removed.")
        
    try:
        feature_list.remove('date')
        feature_list.remove('latitude')
        feature_list.remove('longitude')
        logger.debug("Removed index columns from feature list.")
    except ValueError:
        logger.error("Index columns ('date', 'latitude', 'longitude') not found in feature list, or could not be removed.")
    
    # Training set from all but the target variable
    X = data[feature_list]
    # Testing set is the target variable
    y = data[[target_param]]
    
    logger.info("Splitting data into training and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    if apply_smote:
        logger.info("Applying SMOTE to training data...")
        logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        X_train, y_train = self.preprocessor.apply_smote(X=X_train, y=y_train)
      
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






