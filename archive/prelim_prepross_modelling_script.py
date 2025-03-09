import os
import pandas as pd
import numpy as np
import logging
import joblib  # For saving and loading objects
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
)
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTEENN
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import regularizers
 
# ---------------------------
# 1. Setup Logging
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("model_training.log"),
        logging.StreamHandler()
    ]
)
 
# ---------------------------
# 2. Define Paths
# ---------------------------
input_directory = r"C:\Users\azhari\Desktop\inno"
processed_data_path = os.path.join(input_directory, "processed_data.joblib")
balanced_data_path = os.path.join(input_directory, "balanced_data.joblib")
model_save_path = os.path.join(input_directory, "neural_network_model.h5")
 
# ---------------------------
# 3. Load or Preprocess Data
# ---------------------------
def load_or_preprocess_data():
    if os.path.exists(balanced_data_path):
        logging.info("Loading preprocessed and balanced data from disk...")
        X_train_balanced, X_test, y_train_balanced, y_test = joblib.load(balanced_data_path)
    else:
        if os.path.exists(processed_data_path):
            logging.info("Loading preprocessed data from disk...")
            all_data = joblib.load(processed_data_path)
        else:
            logging.info("Reading CSV files from input directory...")
            all_csv_files = [
                file for file in os.listdir(input_directory) if file.endswith(".csv")
            ]
 
            if not all_csv_files:
                logging.error("No CSV files found in the input directory.")
                raise FileNotFoundError("No CSV files found in the input directory.")
 
            # Initialize an empty DataFrame to store all data
            all_data = pd.DataFrame()
 
            # Concatenate all CSVs into a single DataFrame
            for csv_file in all_csv_files:
                file_path = os.path.join(input_directory, csv_file)
                try:
                    data = pd.read_csv(file_path)
                    all_data = pd.concat([all_data, data], ignore_index=True)
                    logging.info(f"Loaded {csv_file} with shape {data.shape}.")
                except Exception as e:
                    logging.error(f"Error reading {csv_file}: {e}")
 
            logging.info(f"Data concatenation complete. Total records: {len(all_data)}")
 
            # Impute missing values for initial numeric features
            logging.info("Imputing missing values for initial numeric features...")
            numeric_features_initial = [
                "t2m",
                "d2m",
                "u10",
                "v10",
                "sp",
                "tp",
                "latitude",
                "longitude",
                "is_fire_day",
            ]
 
            imputer = SimpleImputer(strategy="mean")
            all_data[numeric_features_initial] = imputer.fit_transform(
                all_data[numeric_features_initial]
            )
 
            # Feature engineering - Extract month and season from 'date' column
            logging.info("Performing feature engineering on 'date' column...")
            all_data["date"] = pd.to_datetime(all_data["date"], errors="coerce")
            all_data["month"] = all_data["date"].dt.month
            all_data["season"] = all_data["month"].map(
                {
                    12: "Winter",
                    1: "Winter",
                    2: "Winter",
                    3: "Spring",
                    4: "Spring",
                    5: "Spring",
                    6: "Summer",
                    7: "Summer",
                    8: "Summer",
                    9: "Fall",
                    10: "Fall",
                    11: "Fall",
                }
            )
 
            # One-hot encode 'season'
            all_data = pd.get_dummies(all_data, columns=["season"], drop_first=True)
 
            # Calculate relative humidity and atmospheric dryness
            logging.info("Calculating relative humidity and atmospheric dryness...")
            # Relative Humidity calculation
            e_t = np.exp(
                (17.625 * (all_data["t2m"] - 273.15))
                / (243.04 + (all_data["t2m"] - 273.15))
            )
            e_d = np.exp(
                (17.625 * (all_data["d2m"] - 273.15))
                / (243.04 + (all_data["d2m"] - 273.15))
            )
            all_data["relative_humidity"] = 100 * (e_d / e_t)
 
            # Atmospheric Dryness calculation
            all_data["atmospheric_dryness"] = (
                all_data["t2m"] - all_data[""]
            ).astype(float)
 
            # Create interaction terms
            logging.info("Creating interaction terms...")
            all_data["temp_humidity_interaction"] = (
                all_data["t2m"] * all_data["relative_humidity"]
            )
 
            # Cluster geographic coordinates to capture spatial patterns
            logging.info("Clustering geographic coordinates...")
            kmeans = KMeans(n_clusters=10, random_state=42)
            all_data["region"] = kmeans.fit_predict(
                all_data[["latitude", "longitude"]]
            )
 
            # One-hot encode the 'region' feature
            all_data = pd.get_dummies(all_data, columns=["region"], drop_first=True)
 
            # Create lag features to capture temporal dependencies
            logging.info("Creating lag features...")
            # Ensure data is sorted by date
            all_data.sort_values("date", inplace=True)
 
            # Create lag features for the past 3 days
            lag_features = ["t2m", "d2m", "relative_humidity"]
            for feature in lag_features:
                for lag in range(1, 4):
                    all_data[f"{feature}_lag_{lag}"] = all_data[feature].shift(lag)
 
            # Drop rows with NaN values resulting from lagging
            all_data.dropna(inplace=True)
 
            # Save preprocessed data to disk
            logging.info("Saving preprocessed data to disk...")
            joblib.dump(all_data, processed_data_path)
            logging.info(f"Preprocessed data saved to {processed_data_path}.")
 
        # Defining features and target
        logging.info("Defining features and target variable...")
        target = "is_fire_day"
 
        # Exclude non-numeric columns from features
        excluded_features = ["valid_time", "date", "expver"]
        features = [
            col
            for col in all_data.columns
            if col not in [target] + excluded_features
        ]
 
        logging.info(f"Features before preprocessing: {features}")
 
        X = all_data[features]
        y = all_data[target]
 
        # Impute any remaining missing values in numeric features
        logging.info("Imputing any remaining missing values in numeric features...")
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        imputer = SimpleImputer(strategy="mean")
        X[numeric_features] = imputer.fit_transform(X[numeric_features])
 
        # Standardize numeric features
        logging.info("Scaling numeric features using StandardScaler...")
        scaler = StandardScaler()
        X[numeric_features] = scaler.fit_transform(X[numeric_features])
 
        # Split the data into training and test sets
        logging.info("Splitting data into training and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
 
        # Handle imbalance in the dataset using SMOTEENN
        logging.info("Handling data imbalance using SMOTEENN...")
        smote_enn = SMOTEENN(random_state=42)
        X_train_balanced, y_train_balanced = smote_enn.fit_resample(X_train, y_train)
        logging.info(f"Training data size after SMOTEENN: {X_train_balanced.shape[0]} samples")
 
        # Save balanced data to disk
        logging.info("Saving balanced data to disk...")
        joblib.dump(
            (X_train_balanced, X_test, y_train_balanced, y_test),
            balanced_data_path,
        )
        logging.info(f"Balanced data saved to {balanced_data_path}.")
 
    return X_train_balanced, X_test, y_train_balanced, y_test
 
# ---------------------------
# 4. Build and Train the Neural Network
# ---------------------------
def build_and_train_model(X_train, y_train, X_val, y_val):
    # Identify numeric features
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    input_dim = X_train.shape[1]
 
    # Define the model architecture
    model = Sequential([
        Dense(128, input_dim=input_dim, activation='relu',
              kernel_regularizer=regularizers.l2(0.001)),
        Dropout(0.5),
        Dense(64, activation='relu',
              kernel_regularizer=regularizers.l2(0.001)),
        Dropout(0.5),
        Dense(32, activation='relu',
              kernel_regularizer=regularizers.l2(0.001)),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Binary classification
    ])
 
    # Define a custom F1 score metric
    def f1_score_metric(y_true, y_pred):
        y_pred_binary = tf.round(y_pred)
        tp = tf.reduce_sum(tf.cast(y_true * y_pred_binary, 'float'), axis=0)
        fp = tf.reduce_sum(tf.cast((1 - y_true) * y_pred_binary, 'float'), axis=0)
        fn = tf.reduce_sum(tf.cast(y_true * (1 - y_pred_binary), 'float'), axis=0)
 
        p = tp / (tp + fp + tf.keras.backend.epsilon())
        r = tp / (tp + fn + tf.keras.backend.epsilon())
 
        f1 = 2*p*r / (p + r + tf.keras.backend.epsilon())
        return f1
 
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['AUC', 'Precision', 'Recall', f1_score_metric]
    )
 
    logging.info("Neural network model architecture defined and compiled.")
 
    # Define EarlyStopping and ModelCheckpoint callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
 
    checkpoint = ModelCheckpoint(
        filepath=model_save_path,
        monitor='val_f1_score_metric',
        mode='max',
        save_best_only=True,
        verbose=1
    )
 
    # Calculate class weights to handle any remaining imbalance
    logging.info("Calculating class weights for training...")
    counter = Counter(y_train)
    total = len(y_train)
    class_weight = {0: 1.0, 1: (total / counter[1])}
 
    logging.info(f"Class weights: {class_weight}")
 
    # Train the model
    logging.info("Starting model training...")
    history = model.fit(
        X_train,
        y_train,
        epochs=100,
        batch_size=1024,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, checkpoint],
        class_weight=class_weight,
        verbose=1
    )
 
    logging.info("Model training completed.")
 
    return model, history
 
# ---------------------------
# 5. Evaluate the Model
# ---------------------------
def evaluate_model(model, X_test, y_test):
    logging.info("Evaluating the model on the test set...")
 
    # Predict probabilities
    y_pred_prob = model.predict(X_test).ravel()
 
    # Predict classes based on a default threshold of 0.5
    y_pred = (y_pred_prob >= 0.5).astype(int)
 
    # Calculate evaluation metrics
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
 
    logging.info(f"ROC AUC Score: {roc_auc:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"F1 Score: {f1:.4f}")
 
    # Optimize the classification threshold based on F1 Score
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-6)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
 
    # Apply the optimal threshold
    y_pred_optimal = (y_pred_prob >= optimal_threshold).astype(int)
 
    # Recalculate metrics
    precision_opt = precision_score(y_test, y_pred_optimal)
    recall_opt = recall_score(y_test, y_pred_optimal)
    f1_opt = f1_score(y_test, y_pred_optimal)
 
    logging.info(f"Optimal Threshold: {optimal_threshold:.4f}")
    logging.info(f"Optimized Precision: {precision_opt:.4f}")
    logging.info(f"Optimized Recall: {recall_opt:.4f}")
    logging.info(f"Optimized F1 Score: {f1_opt:.4f}")
 
    # Final evaluation summary
    print("\nFinal Evaluation Metrics:")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print(f"Precision: {precision_opt:.4f}")
    print(f"Recall: {recall_opt:.4f}")
    print(f"F1 Score: {f1_opt:.4f}")
 
# ---------------------------
# 6. Main Execution Flow
# ---------------------------
def main():
    try:
        # Load or preprocess data
        X_train_balanced, X_test, y_train_balanced, y_test = load_or_preprocess_data()
 
        # Further split training data into training and validation sets
        logging.info("Splitting balanced training data into training and validation sets...")
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_balanced, y_train_balanced, test_size=0.2, random_state=42, stratify=y_train_balanced
        )
        logging.info(f"Training set: {X_train.shape[0]} samples")
        logging.info(f"Validation set: {X_val.shape[0]} samples")
 
        # Build and train the model
        model, history = build_and_train_model(X_train, y_train, X_val, y_val)
 
        # Load the best model saved by ModelCheckpoint
        if os.path.exists(model_save_path):
            logging.info(f"Loading the best model from {model_save_path}...")
            best_model = tf.keras.models.load_model(
                model_save_path,
                custom_objects={'f1_score_metric': lambda y_true, y_pred: f1_score_metric(y_true, y_pred)}
            )
        else:
            best_model = model
            logging.warning(f"Best model checkpoint not found at {model_save_path}. Using the latest model.")
 
        # Evaluate the model
        evaluate_model(best_model, X_test, y_test)
 
    except Exception as e:
        logging.exception(f"An error occurred during the model training process: {e}")
        print(f"An error occurred: {e}")
 
if __name__ == "__main__":
    main()