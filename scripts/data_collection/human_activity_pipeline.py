# human_activity_pipeline.py

import requests
import logging
import pandas as pd
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HumanActivityPipeline:
    """
    Pipeline to fetch human activity data from the Overpass API, focusing on:
      - railways
      - industrial land (landuse=industrial)
      - power lines (power=line)

    Now we do ONE bounding box per monthly batch to reduce calls/caching:
      - We compute the bounding box from all lat/lons in that batch.
      - We store a single CSV in the cache like 'osm_200601.csv'.
      - We parse the Overpass result and store all found OSM features so we can
        quickly look up which rail/power/industrial features are near each lat/lon.
    """

    def __init__(self):
        self.overpass_url = "https://overpass-api.de/api/interpreter"
        self.cache_dir = "osm_cache"
        os.makedirs(self.cache_dir, exist_ok=True)

        logger.info("HumanActivityPipeline initialized.")

    def _cache_filename(self, period_key):
        """
        Cache file is monthly-based, e.g., 'osm_200601.csv' for Jan 2006,
        storing all OSM features in that bounding box.
        """
        return os.path.join(self.cache_dir, f"osm_{period_key}.csv")

    def _save_cache(self, filename, df):
        df.to_csv(filename, index=False)

    def _load_cache(self, filename):
        if os.path.exists(filename):
            return pd.read_csv(filename)
        return None

    def _fetch_overpass_data(self, south, west, north, east):
        """
        Single Overpass call for bounding box: (south, west, north, east).
        Query railways, industrial landuse, power=line.
        """
        bbox = f"{south},{west},{north},{east}"

        overpass_query = f"""
        [out:json];
        (
          node["railway"]({bbox});
          way["railway"]({bbox});
          node["landuse"="industrial"]({bbox});
          way["landuse"="industrial"]({bbox});
          node["power"="line"]({bbox});
          way["power"="line"]({bbox});
        );
        out center;
        """

        try:
            response = requests.get(self.overpass_url, params={'data': overpass_query}, timeout=60)
            response.raise_for_status()
            data = response.json()
            elements = data.get('elements', [])
            return pd.DataFrame(elements)
        except requests.exceptions.RequestException as e:
            logger.error(f"Overpass API request failed: {e}")
            return pd.DataFrame()

    def fetch_human_activity_monthly(self, monthly_df, period_key):
        """
        For the entire monthly DataFrame (with columns lat/lon),
        compute bounding box, check cache, fetch Overpass once,
        then for each point, count how many rail/power/industrial features
        are "close" to it within a small radius.

        Alternatively, if you want to do an EXACT lat/lon approach, you can do
        geometry-based checks, but here we do a simple tag-based approach:
           - We'll store OSM elements as raw, then do a naive lat/lon proximity
             to each point (like a 0.05 deg tolerance).

        If this is too coarse, you can do row-level bounding boxes for each point
        but it will defeat the purpose of monthly caching.
        """
        # 1) bounding box from monthly_df lat/lon
        lat_min = monthly_df['latitude'].min()
        lat_max = monthly_df['latitude'].max()
        lon_min = monthly_df['longitude'].min()
        lon_max = monthly_df['longitude'].max()

        # 2) see if we have a monthly cache
        cache_file = self._cache_filename(period_key)
        osm_df = self._load_cache(cache_file)
        if osm_df is None:
            logger.info(f"No monthly cache found for {period_key}. Querying Overpass with bounding box.")
            osm_df = self._fetch_overpass_data(lat_min, lon_min, lat_max, lon_max)
            if not osm_df.empty:
                self._save_cache(cache_file, osm_df)
                logger.info(f"Cached monthly Overpass data -> {cache_file}")
            else:
                logger.warning(f"No OSM data returned for this bounding box: lat=[{lat_min}, {lat_max}], lon=[{lon_min}, {lon_max}]")
                # We'll store an empty df so we don't re-query
                empty_df = pd.DataFrame(columns=['id','type','lat','lon','tags'])
                self._save_cache(cache_file, empty_df)
                return monthly_df  # no data, just return monthly_df as-is
        else:
            logger.info(f"Loaded monthly OSM data from {cache_file}")

        # 3) parse OSM features: we keep them as raw
        # Then for each row in monthly_df, we do a small lat-lon check or skip it entirely.
        # Because we've a single bounding box for the entire month, the user might want
        # a smaller local tolerance to decide if there's a railway/power/industrial "nearby."

        # We'll define a small tolerance (0.05 deg) so we only count OSM features that are
        # "close" to each row's lat/lon. Adjust as you see fit.
        osm_df = osm_df.dropna(subset=['lat','lon'])  # keep valid positions

        def has_tag(r, key, val):
            t = r.get('tags', {})
            return (isinstance(t, dict) and t.get(key) == val)

        # Let's store for each OSM element:
        #   'feature_type' in { 'railway', 'industrial', 'power_line' }
        # We'll keep 'lat','lon' for each, so we can do distance checks.
        processed_elements = []
        for _, elem in osm_df.iterrows():
            tags = elem.get('tags', {})
            if not isinstance(tags, dict):
                continue

            lat_ = elem.get('lat')
            lon_ = elem.get('lon')

            # check if it's a railway
            if 'railway' in tags:
                processed_elements.append({
                    'lat': lat_,
                    'lon': lon_,
                    'feature_type': 'railway'
                })
            elif tags.get('landuse') == 'industrial':
                processed_elements.append({
                    'lat': lat_,
                    'lon': lon_,
                    'feature_type': 'industrial'
                })
            elif tags.get('power') == 'line':
                processed_elements.append({
                    'lat': lat_,
                    'lon': lon_,
                    'feature_type': 'power_line'
                })

        if not processed_elements:
            logger.warning(f"Overpass bounding box found 0 relevant elements for {period_key}.")
            return monthly_df

        processed_osm = pd.DataFrame(processed_elements)
        logger.info(f"Parsed Overpass data for {period_key}: total OSM rows={len(osm_df)}, relevant features={len(processed_osm)}")

        # We'll define a lat/lon tolerance for "nearby" (like 0.05 deg).
        # Each monthly row checks how many rail/industrial/power lines are near it
        #  in that bounding box.
        def count_features_for_row(row):
            lat0, lon0 = row['latitude'], row['longitude']
            tolerance = 0.05
            # filter processed_osm to only those within ±0.05 deg
            nearby = processed_osm[
                (processed_osm['lat'].between(lat0 - tolerance, lat0 + tolerance)) &
                (processed_osm['lon'].between(lon0 - tolerance, lon0 + tolerance))
            ]
            # count each feature type
            railway_count = (nearby['feature_type'] == 'railway').sum()
            industrial_count = (nearby['feature_type'] == 'industrial').sum()
            power_line_count = (nearby['feature_type'] == 'power_line').sum()
            return pd.Series([railway_count, industrial_count, power_line_count])

        # We'll do a single pass and store these counts in columns
        # We'll also track how many had non-zero features to reduce logging spam
        found_features = 0
        total_rows = len(monthly_df)

        # We'll do an apply
        monthly_df[['railway_count','industrial_count','power_line_count']] = monthly_df.apply(
            lambda row: count_features_for_row(row), axis=1
        )

        found_features = ( (monthly_df['railway_count']>0) | 
                           (monthly_df['industrial_count']>0) |
                           (monthly_df['power_line_count']>0) ).sum()

        logger.info(f"[{period_key}] OSM results: {found_features} / {total_rows} rows have at least one OSM feature near them.")
        return monthly_df
