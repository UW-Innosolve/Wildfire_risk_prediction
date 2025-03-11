from feat_engineer import FeatEngineer
from preprocessor import Preprocessor

class FbDataset(FeatEngineer, Preprocessor):
  def __init__(self, raw_data_df):
    self.raw = raw_data_df
    self.features_list = []
    
  # NOTE: THESE ARE FEATURE DEFAULTS, DO NOT EDIT THESE.
  ## Instead, override the parameters using the config_features method
  ## to create an instance with a custom features configuration.
  
  # config_features
  #   raw_params_feats: list of raw parameters to be included as features
  #   eng_feats: list of engineered features to be included
  
  def config_features(self, raw_params_feats, eng_feats):
    self.features_list = dict()
    
    self.raw_param_feats = ['latitute', 'longitude', 
                            '10u',	'10v', '2d', '2t',  # wind, dewpoint, temperature
                            'cl', # Lake cover              
                            'cvh', 'cvl', # Low vegetation cover, high vegetation cover
                            'fal', # Forecast albedo
                            'lai_hv',	'lai_lv', # Leaf area index high vegetation, low vegetation
                            'lsm', # Land-sea mask
                            'slt', # Soil type
                            'sp',	# Surface pressure
                            'src', # Skin reservoir content
                            'stl1',	'stl2',	'stl3',	'stl4', # Soil temperature levels (0-7cm, 7-28cm, 28-100cm, 100-289cm)
                            'swvl1',	'swvl2',	'swvl3',	'swvl4', # Soil water volume levels (0-7cm, 7-28cm, 28-100cm, 100-289cm)
                            'tvh',	'tvl', # High vegetation type, low vegetation type (Categorical)
                            'z', # Geopotential (proportional to elevation)
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
    
    self.eng_feats =          [   # CWFDRS Fire weather indices
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
               
  def gen_features(self):
    pass
    
  # def process(self):
  #   pass
  #    # Definition of features dictionary
  #   #  key: feature name
  #   #  valueL:
    
  #   self.numeric_features = []
    
  #   self.catgorical_features = []