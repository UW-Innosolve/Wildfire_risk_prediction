# %% [markdown]
# # Wildfire Prediction Data EDA Notebook
# 
# **Overview:**  
# This notebook aggregates CSV files from our wildfire prediction pipeline and performs an extensive EDA.
# The parameters includes the following columns (headers):
# 
# - **Temporal & Spatial:** `date`, `latitude`, `longitude`
# - **Weather / Environmental Features:** `10u`, `10v`, `2d`, `2t`, `cl`, `cvh`, `cvl`, `fal`, `lai_hv`, `lai_lv`, `lsm`, `slt`, `sp`, `src`, `stl1`, `stl2`, `stl3`, `stl4`, `swvl2`, `swvl3`, `swvl4`, `tvh`, `tvl`, `z`, `e`, `pev`, `slhf`, `sshf`, `ssr`, `ssrd`, `str`, `strd`, `tp`
# - **Target Variable:** `is_fire_day`
# 
# Our goals are to understand data quality, identify patterns, and prepare for feature engineering and model development.

# %% [code]
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from datetime import datetime
import sklearn

# Set default plot style
sns.set(style="whitegrid")
%matplotlib inline

# %% [markdown]
# ## 1. Data Aggregation
# 
# We will aggregate all CSV files from a specified directory into a single DataFrame.

# %% [code]
# Define the directory containing the CSV files
input_directory = r"C:\Users\ibuaz\OneDrive\Desktop\inno\EDA"  # Update path as needed

# Use glob to find all CSV files
csv_files = glob.glob(os.path.join(input_directory, "*.csv"))

# Create a list to hold DataFrames
dataframes = []
for file in csv_files:
    try:
        df_temp = pd.read_csv(file)
        dataframes.append(df_temp)
        print(f"Loaded {os.path.basename(file)} with shape {df_temp.shape}")
    except Exception as e:
        print(f"Error loading {file}: {e}")

# Combine all DataFrames
if dataframes:
    df = pd.concat(dataframes, ignore_index=True)
else:
    raise ValueError("No CSV files found.")

print("Aggregated DataFrame shape:", df.shape)

# %% [markdown]
# ## 2. Print and Verify Headers
# 
# Let's print out all headers (column names) to confirm our parameters structure.

# %% [code]
print("Headers in the aggregated parameters:")
for col in df.columns:
    print(col)
    
# %% [markdown]
# ## 3. Data Overview and Cleaning
# 
# We will inspect the parameters's structure, check for missing values, and view basic statistics.

# %% [code]
# Display parameters information
print("Dataset Info:")
df.info()

# Display summary statistics for numerical features
print("\nSummary Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values per Column:")
print(df.isnull().sum())
    
# %% [markdown]
# ## 4. Convert and Validate Date Column
# 
# Ensure the `date` column is in datetime format and print the date range.

# %% [code]
if df['date'].dtype == object:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
print("Date range:", df['date'].min(), "to", df['date'].max())

# %% [markdown]
# ## 5. Distribution Analysis of Weather Features
# 
# Plot histograms and KDE plots for key weather/environmental features.

# %% [code]
# List of continuous weather features (adjust based on knowledge of the variables)
weather_features = ['10u', '10v', '2d', '2t', 'cl', 'cvh', 'cvl', 'fal', 
                    'lai_hv', 'lai_lv', 'lsm', 'slt', 'sp', 'src', 
                    'stl1', 'stl2', 'stl3', 'stl4', 'swvl2', 'swvl3', 'swvl4', 
                    'tvh', 'tvl', 'z', 'e', 'pev', 'slhf', 'sshf', 'ssr', 'ssrd', 'str', 'strd', 'tp']

for feature in weather_features:
    if feature in df.columns:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[feature].dropna(), kde=True, bins=30, color='skyblue')
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.show()
        
# %% [markdown]
# ## 6. Outlier Detection
# 
# Use boxplots to visualize potential outliers in the continuous features.

# %% [code]
for feature in weather_features:
    if feature in df.columns:
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=df[feature])
        plt.title(f"Boxplot of {feature}")
        plt.xlabel(feature)
        plt.show()

# %% [markdown]
# ## 7. Target Variable Analysis
# 
# Analyze the distribution of the target variable (`is_fire_day`).

# %% [code]
plt.figure(figsize=(6, 4))
sns.countplot(x='is_fire_day', data=df, palette='viridis')
plt.title("Distribution of Fire vs Non-Fire Days")
plt.xlabel("Fire Day (1 = Yes, 0 = No)")
plt.ylabel("Count")
plt.show()

# %% [markdown]
# ## 8. Correlation Analysis
# 
# Generate and visualize a correlation matrix to examine relationships among features.

# %% [code]
plt.figure(figsize=(14, 12))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix of Features")
plt.show()

# %% [markdown]
# ## 9. Temporal Analysis
# 
# Examine how features and fire occurrences change over time.

# %% [code]
# Time series of a key weather feature (e.g., '2t' - 2m temperature)
plt.figure(figsize=(12, 6))
sns.lineplot(x='date', y='2t', data=df)
plt.title("2m Temperature Trend Over Time")
plt.xlabel("Date")
plt.ylabel("Temperature (K)")
plt.show()

# Extract month from date
df['month'] = df['date'].dt.month

# Fire occurrence by month
plt.figure(figsize=(8, 5))
sns.countplot(x='month', hue='is_fire_day', data=df, palette='viridis')
plt.title("Monthly Fire Occurrence")
plt.xlabel("Month")
plt.ylabel("Count")
plt.show()

# %% [markdown]
# ## 10. Spatial Analysis
# 
# Visualize the spatial distribution of data points and fire occurrences.

# %% [code]
plt.figure(figsize=(10, 8))
sns.scatterplot(x='longitude', y='latitude', hue='is_fire_day', data=df, palette={0: 'blue', 1: 'red'}, alpha=0.7)
plt.title("Spatial Distribution of Fire Occurrence")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

# %% [markdown]
# ## 11. Interactive Map Visualization
# 
# Build an interactive map with Folium to explore spatial patterns in fire occurrences.

# %% [code]
import folium
from folium.plugins import MarkerCluster

# Create a base map centered on Alberta
m = folium.Map(location=[55, -115], zoom_start=5)
marker_cluster = MarkerCluster().add_to(m)

# Add data points to the map with different colors based on fire occurrence
for idx, row in df.iterrows():
    color = 'red' if row['is_fire_day'] == 1 else 'blue'
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=3,
        color=color,
        fill=True,
        fill_opacity=0.7
    ).add_to(marker_cluster)

# Save and display the map
m.save("interactive_fire_map.html")
m

# %% [markdown]
# ## 12. Feature Importance Analysis
# 
# Use a Random Forest model to evaluate which features are most important for predicting fire days.

# %% [code]
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Define features for the model: use weather and spatial features; adjust as needed
#model_features = ['2t', '2d', '10u', '10v', 'sp', 'tp', 'relative_humidity', 'atmospheric_dryness', 'latitude', 'longitude', 'month']
model_features = ['10u', '10v', '2d', '2t', 'cl', 'cvh', 'cvl', 'fal', 
                    'lai_hv', 'lai_lv', 'lsm', 'slt', 'sp', 'src', 
                    'stl1', 'stl2', 'stl3', 'stl4', 'swvl2', 'swvl3', 'swvl4', 
                    'tvh', 'tvl', 'z', 'e', 'pev', 'slhf', 'sshf', 'ssr', 'ssrd', 'str', 'strd', 'tp']
X = df[model_features].dropna()
y = df.loc[X.index, 'is_fire_day']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)

importances = rf_model.feature_importances_
feature_importance_df = pd.DataFrame({'feature': X_train.columns, 'importance': importances}).sort_values(by='importance', ascending=False)
print("Feature Importances:")
print(feature_importance_df)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance_df, palette='viridis')
plt.title("Feature Importances from Random Forest")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()

# %% [markdown]
# ## 13. Data Balancing Assessment
# 
# Evaluate the class imbalance and demonstrate resampling using SMOTE.

# %% [code]
from imblearn.over_sampling import SMOTE

print("Class distribution before SMOTE:")
print(y_train.value_counts())

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

print("Class distribution after SMOTE:")
print(pd.Series(y_res).value_counts())
# %% [markdown]
