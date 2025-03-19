from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import logging

def mm_scale_df(df):
    """
    Scale the input DataFrame using MinMaxScaler.
    """
    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()
    # Fit and transform the input DataFrame
    df_scaled = scaler.fit_transform(df)
    # Convert the scaled data back to a DataFrame
    df_scaled = pd.DataFrame(df_scaled, columns=df.columns, index=df.index)
    return df_scaled
  
  