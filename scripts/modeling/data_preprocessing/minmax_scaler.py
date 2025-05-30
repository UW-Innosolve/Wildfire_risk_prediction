from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import logging

def mm_scale_df(df):
    """
    Scale the input DataFrame using MinMaxScaler.
    """
    # Initialize the MinMaxScaler
    df_returned = df
    scaler = MinMaxScaler()
    # Fit and transform the input DataFrame
    df_scaled = scaler.fit_transform(df[3:])
    # Convert the scaled data back to a DataFrame
    df_returned[3:] = pd.DataFrame(df_scaled, columns=df.columns[3:], index=df.index)
    return df_returned
  
  