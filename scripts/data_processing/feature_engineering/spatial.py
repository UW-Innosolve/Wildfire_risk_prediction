from sklearn.cluster import KMeans
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FbSpatialFeatures():
    def __init__(self, raw_data_df):
        self.raw_data = raw_data_df.copy()
        # self.raw_data['lat'] = raw_data_df['latitude'].astype(float)
        # self.raw_data['lon'] = raw_data_df['longitude'].astype(float)
        self.spatial_features = self.raw_data[['date', 'latitude', 'longitude']]

    def kmeans_cluster(self, n_clusters=12):
        """
        KMeans clustering on lat, lon columns.
        """
        random_state= 42
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        clustering = pd.DataFrame()
        clustering[f'clusters_{n_clusters}'] = kmeans.fit_predict(self.raw_data[['latitude', 'longitude']])
        self.spatial_features[f'clusters_{n_clusters}'] = clustering[f'clusters_{n_clusters}']
    
    def get_features(self):
        return self.spatial_features
      