from feat_engineer import FeatEngineer
from preprocessor import Preprocessor

class FbDataset(FeatEngineer, Preprocessor):
  def __init__(self, raw_data_df):
    self.raw = raw_data_df
    
  
  def config_features(self, raw_params, numeric_features, categorical_features):
    self.raw_params = ['latitute', 'longitude',
                       '10u',	'10v', '2d', '2t',
                       'cl',	'cvh',	'cvl',	'fal',
                       'lai_hv',	'lai_lv',
                       'lsm',	'slt',	'sp',	'src',
                       'stl1',	'stl2',	'stl3',	'stl4',
                       'swvl1',	'swvl2',	'swvl3',	'swvl4',
                       'tvh',	'tvl',
                      #  'z', elevation omitted in favour of 
                       'e',	'pev',
                       'slhf',	'sshf',	'ssr',	'ssrd',
                       'str',	'strd',	'tp',
                       'is_fire_day', # NOTE: This is the target variable
                       'lightning_count', 'absv_strength_sum',	'multiplicity_sum',
                       'railway_count',	'power_line_count',	'highway_count',
                       'aeroway_count',	'waterway_count']
    
    self.numeric_features = ['10u',	'10v', '2d', '2t',
                             'cl',          # Lake cover
                             'lsm',         # Land-sea mask
                              'z',           # Geopotential (proportional to elevation)
                              'slt']
                      
  def gen_features(self):
    pass
    
  def process(self):
    pass