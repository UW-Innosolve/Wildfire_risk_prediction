import pandas as pd
import numpy as np
import datetime as dt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FbTemporalFeatures():
  def __init__(self, raw_data_df):
    self.raw_data = raw_data_df
    self.temporal_features = pd.DataFrame()
    self.date_df = pd.to_datetime(self.raw_data["date"], errors="coerce")
    self.year_df = self.date_df.year
    self.month_df = self.date_df.month
    self.day_of_month = self.date_df.day
    
    
  def seasonal(self):
    '''
    Extracts seasonal feature (catagorical) based on northern hemisphere solstices and equinoxes
      Winter: Dec 21 - Mar 19
      Spring: Mar 20 - Jun 20
      Summer: Jun 21 - Sep 21
      Autumn (Fall): Sep 22 - Dec 20
    '''
    seasons = []

    winter_start = pd.Timestamp("12-21")
    spring_start = pd.Timestamp("03-20")
    summer_start = pd.Timestamp("06-21")
    autumn_start = pd.Timestamp("09-22")

    for date in self.date_df:
      year = date.year
      date_without_year = pd.Timestamp(f"{date.month}-{date.day}")

      if date_without_year >= winter_start or date_without_year < spring_start:
          season = "Winter"
      elif spring_start <= date_without_year < summer_start:
          season = "Spring"
      elif summer_start <= date_without_year < autumn_start:
          season = "Summer"
      else:
          season = "Autumn"
          
      seasons.append(season)
      
    self.temporal_features['season'] = seasons
      
      
  def fire_seasonal(self):
    '''
    Extracts fire season catagory feature
      off_season: Jan 1 - Mar 1 and Oct 1 - Oct 31
      on_season: Mar 2 - April 30 and Sep 1 - Sep 31
      core_season: May 1 - Aug 31
    '''
    
    fire_seasons = []

    off_season_start_1 = pd.Timestamp("01-01")
    off_season_end_1 = pd.Timestamp("03-01")
    off_season_start_2 = pd.Timestamp("10-01")
    off_season_end_2 = pd.Timestamp("12-31")

    on_season_start_1 = pd.Timestamp("03-02")
    on_season_end_1 = pd.Timestamp("04-30")
    on_season_start_2 = pd.Timestamp("09-01")
    on_season_end_2 = pd.Timestamp("09-30")

    core_season_start = pd.Timestamp("05-01")
    core_season_end = pd.Timestamp("08-31")

    for date in self.date_df:
      date_without_year = pd.Timestamp(f"{date.month}-{date.day}")

      if (off_season_start_1 <= date_without_year <= off_season_end_1) or (off_season_start_2 <= date_without_year <= off_season_end_2):
        fire_season = "off_season"
      elif (on_season_start_1 <= date_without_year <= on_season_end_1) or (on_season_start_2 <= date_without_year <= on_season_end_2):
        fire_season = "on_season"
      elif core_season_start <= date_without_year <= core_season_end:
        fire_season = "core_season"
      else:
        logger.error("Unsorted date in fire season feature extraction")
        return

      fire_seasons.append(fire_season)

    self.temporal_features['fire_season'] = fire_seasons
    
    
  # NOTE: seasonal features are catagorical, must be onehotted later.
  def features(self, seasonal=True, fire_season=True):
    if seasonal:
      self.seasonal()
    if fire_season:
      self.fire_seasonal()
    
    return self.temporal_features
      
