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
        
        self.data_idx = pd.DataFrame(self.data, columns = ['date', 'latitude', 'longitude', 'is_fire_day'])
        self.data_idx.set_index(['date', 'latitude', 'longitude'], inplace=True)

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
        data_idx = self.data_idx # start with the date, latitude, longitude index
        data_ss = self.feature_scaler_ss.fit_transform(feature_df)
        data_ss = pd.merge(data_idx,
                           pd.DataFrame(data_ss, columns=feature_df.columns, index=self.data_idx.index),
                           left_index=True, right_index=True)
        return data_ss
    
    def scale_features_mm(self, feature_df):
        """
        Scale features using MinMax.
        """
        data_idx = self.data_idx # start with the date, latitude, longitude index
        data_mm = self.feature_scaler_minmax.fit_transform(feature_df)
        data_mm = pd.merge(data_idx,
                           pd.DataFrame(data_mm, columns=feature_df.columns, index=self.data_idx.index),
                           left_index=True, right_index=True)
        return data_mm
        # feature_list = feature_df.columns
        # data_mm = self.feature_scaler_minmax.fit_transform(feature_df)
        # data_mm = pd.DataFrame(data_mm, columns=feature_list, index=self.data_idx)
        # return data_mm
    
    def onehot_cat_features(self, feature_df):
        """
        One-hot encode categorical features.
        """
        feature_list = feature_df.columns
        data_idx = self.data_idx # start with the date, latitude, longitude index
        data_onehot = pd.get_dummies(feature_df, columns=feature_list, drop_first=True) # Drop first to avoid multicollinearity
        data_onehot = pd.merge(data_idx, data_onehot,
                               index=self.data_idx.index,
                               left_index=True, right_index=True)
        return data_onehot

    def apply_smote(self, X, y):
        """
        Apply SMOTE to oversample the minority class (fire days).
         - Useful for handling class imbalance, worked pretty well prev iteration.
        """
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
        return X_res, y_res

    def split_data(self, feature_list, target, test_size=0.2, random_state=42, apply_smote=False):
        """
        Split the data into training and testing sets.
         - Optionally apply SMOTE to the training data.
         - Parameters like test_size and random_state can be adjusted.
        """
        X = self.data[feature_list]
        y = self.data[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        if apply_smote:
            X_train, y_train = self.apply_smote(X_train, y_train)
        return X_train, X_test, y_train, y_test
