import os
import glob
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

class Preprocessor:
    def __init__(self, data_dir):
        """
        Initialize the Preprocessor with the directory path containing CSV files.
        """
        self.data_dir = data_dir
        self.data = None
        self.scaler_ss = StandardScaler()  # Standardize features for improved model performance
        self.scale_features_minmax = MinMaxScaler()  # Normalize features for algorithms that require it

    def load_data(self):
        """
        Aggregate all CSV files from the specified directory into a single DataFrame.
        Uses glob to fetch file paths and concatenates them.
        """
        csv_files = glob.glob(os.path.join(self.data_dir, "*.csv"))
        
        print("Found CSV files:", csv_files)
        dfs = []
        for file in csv_files:
            try:
                df_temp = pd.read_csv(file)
                print(f"Loaded {os.path.basename(file)} with shape {df_temp.shape}")
                dfs.append(df_temp)
            except Exception as e:
                print(f"Error loading {file}: {e}")
        if dfs:
            self.data = pd.concat(dfs, ignore_index=True)
            print("Aggregated DataFrame shape:", self.data.shape)
        else:
            raise ValueError("No CSV files found in the specified directory.")
        return self.data

    def clean_data(self):
        """
        Basic cleaning:
         - Convert 'date' column to datetime.
         - Drop rows with missing target ('is_fire_day').
        """
        if 'date' in self.data.columns:
            self.data['date'] = pd.to_datetime(self.data['date'], errors='coerce')
        self.data.dropna(subset=['is_fire_day'], inplace=True)
        return self.data

    def scale_features_ss(self, feature_list):
        """
        Scale features using StandardScaler.
         - This is important for models like KNN and Logistic Regression.
        """
        self.data[feature_list] = self.scaler.fit_transform(self.data[feature_list])
        return self.data
    
    def scale_features_minmax(self, feature_list):
        """
        Scale features using MinMax.
        """
        self.data[feature_list] = self.scale_features_minmax.fit_transform(self.data[feature_list])
        return self.data
    
    def onehot_cat_features(self, feature_list):
        """
        One-hot encode categorical features.
        """
        self.data[feature_list] = pd.get_dummies(self.data[feature_list], columns=feature_list, drop_first=True)
        return self.data

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
