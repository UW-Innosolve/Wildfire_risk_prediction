import sklearn.preprocessing as skp
import numpy as np


class data_preprocessor:
  def __init__(self):
    pass
  
  # norm_col_scale_minmax
  # - data_col: column of data to be normalized
  # - type: 'absval' or 'standard'
  # - return: normalized column between -val and val if 'standard; or 0 and val if 'absval'
  def norm_col_scale_minmax(self, data_col, val, type):
    pass
    
  # norm_col_descretize
  # - data_col: column of data to be descretized in to bins
  # - bins: number of bins to descretize the data into
  # - return: descretized column
  def norm_col_descretize(self, data_col, bins):
    pass
  
  # norm_col_binarize
  # - data_col: column of data to be binarized into 0 or 1
  # - threshold: value to binarize the column
  # - return: binarized column
  def norm_col_binarize(self, data_col, threshold):
    pass
  



  
  
  
  