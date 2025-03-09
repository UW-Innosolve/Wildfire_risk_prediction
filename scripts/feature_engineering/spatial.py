from sklearn.cluster import KMeans

class FbSpatialFeatures():
    def __init__(self, input_df):
        self.df = input_df
        self.df['lat'] = input_df['latitude'].astype(float)
        self.df['lon'] = input_df['longitude'].astype(float)

    def kmeans_cluster(self, n_clusters=12):
        """
        KMeans clustering on lat, lon columns.
        """
        random_state= 42
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.df['cluster'] = kmeans.fit_predict(self.df[['lat', 'lon']])
        return self.df
      