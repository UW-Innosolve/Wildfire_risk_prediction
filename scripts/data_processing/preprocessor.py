import os
import glob
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Preprocessor:
    def __init__(self, raw_data_df):
        """
        Initialize the preprocessor with the raw data Dataframe.
        """
        self.data = raw_data_df # DataFrame to store the aggregated raw data (no feautues or processing)
        logger.info("Preprocessor initialized with df shape: {}".format(self.data.shape))
        self.feature_scaler_ss = StandardScaler()  # Standardize features for improved model performance
        self.feature_scaler_minmax = MinMaxScaler()  # Normalize features for algorithms that require it
        
        self.raw_data_df = raw_data_df.copy()
        self.data_idx = self.raw_data_df[['date', 'latitude', 'longitude']]
        logger.debug(f"Pre-processor index shape: {self.data_idx.shape}")


    def clean_data(self):
        """
        Basic cleaning:
         - Convert 'date' column to datetime.
         - Drop rows with missing target ('is_fire_day').
        """
        logger.info("Cleaning data (converting dates, removing missing target values)...")
        if 'date' in self.data.columns:
            self.data['date'] = pd.to_datetime(self.data['date'], errors='coerce')
        self.data.dropna(subset=['is_fire_day'], inplace=True)
        return self.data


    def scale_features_ss(self, feature_df): 
        """
        Scale features using StandardScaler.
         - This is important for models like KNN and Logistic Regression.
        """
        logger.info(f"Scaling standard scaler features: {feature_df.columns}")
        data_idx = self.data_idx.copy() # start with the date, latitude, longitude index
        data_ss = self.feature_scaler_ss.fit_transform(feature_df)
        data_ss = pd.merge(data_idx,
                           pd.DataFrame(data_ss, columns=feature_df.columns, index=self.data_idx.index),
                           left_index=True, right_index=True)
        return data_ss
    
    
    def scale_features_mm(self, feature_df):
        """
        Scale features using MinMax.
        """
        logger.info(f"Scaling min-max features: {feature_df.columns}")
        data_idx = self.data_idx.copy() # start with the date, latitude, longitude index
        data_mm = self.feature_scaler_minmax.fit_transform(feature_df)
        data_mm = pd.merge(data_idx,
                           pd.DataFrame(data_mm, columns=feature_df.columns, index=self.data_idx.index),
                           left_index=True, right_index=True)
        return data_mm
    
    
    def onehot_cat_features(self, feature_df):
        """
        One-hot encode categorical features while preserving the date, latitude, and longitude index.
        """
        data = self.data_idx.copy()  # Preserve index with date, lat, long
        data = data.sort_values(by=['date', 'latitude', 'longitude'])
        data.to_csv("empty_data_index.csv")
        logger.info(f"One-hot encoding features: {feature_df.columns}")
        
        onehot_sets = []
    
        for col in feature_df.columns:
            logger.info(f"Processing one-hot encoding for column: {col}")
            onehot_cols = pd.get_dummies(feature_df[col], prefix=col, drop_first=True)
            logger.info(f"onehot_cols shape: {onehot_cols.shape}, data shape: {data.shape}")
            onehot_cols_with_idx = pd.concat([data.reset_index(drop=True), onehot_cols.reset_index(drop=True)], axis=1)
            print(onehot_cols_with_idx.head())
            
            nulls_in_onehot_col = onehot_cols_with_idx.isnull().values.any().sum()
            if nulls_in_onehot_col > 0:
                logger.warning(f"Null values found in one-hot encoded column: {col}")
                logger.warning(f"Number of null values: {nulls_in_onehot_col}")
                
            onehot_sets.append(onehot_cols_with_idx)
            
            #NOTE: Uncomment to save one-hot encoded columns to CSV
            onehot_cols.to_csv(f"onehot_cols_{col}.csv")
            onehot_cols_with_idx.to_csv(f"onehot_cols_with_idx_{col}.csv")
    
        df_comb = onehot_sets[0]
        logger.info(f"df_comb (0) shape: {onehot_sets[0].shape}")
        for df in onehot_sets[1:]:
            df_comb = pd.merge(df_comb, df, on=['date', 'latitude', 'longitude'], how='outer')
            logger.info(f"df_comb shape: {df_comb.shape}")
            
        logger.info(f"Data shape after one-hot encoding: {df_comb.shape}")
        
        return df_comb


    def apply_smote(self, X, y):
        """
        Apply SMOTE to oversample the minority class (fire days).
         - Useful for handling class imbalance, worked pretty well prev iteration.
        """
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
        return X_res, y_res

