from sklearn.cluster import KMeans
import pandas as pd

class FbSpatialFeatures():
    def __init__(self, raw_data_df):
        self.raw_data = raw_data_df
        self.raw_data['lat'] = raw_data_df['latitude'].astype(float)
        self.raw_data['lon'] = raw_data_df['longitude'].astype(float)
        self.spatial_features = pd.DataFrame()

    def kmeans_cluster(self, n_clusters=12):
        """
        KMeans clustering on lat, lon columns.
        """
        random_state= 42
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.spatial_features['cluster'] = kmeans.fit_predict(self.df[['lat', 'lon']])
        return self.spatial_features
    
    def features(self):
        return self.kmeans_cluster(self.input_df)
      