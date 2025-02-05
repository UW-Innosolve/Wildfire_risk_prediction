import earthaccess
import pandas as pd
import h5py
import logging
import xarray as xr
import requests
import os
import sys
from earthdata_pipeline.nasa_earthdata_pipeline import NasaEarthdataPipeline as ned

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


static_data_path = "scripts/data_collection/static_datasets"
ned = ned() # Initialize the NASA Earthdata pipeline (uses login credentials from credentials.json)

ab_terrain = ned.earthdata_pull_invar(
        short_name='GLAH06',
        doi='10.5067/ICESAT/GLAS/DATA109',
        daac='NSIDC',
        bounding_box=(-120, 49, -110, 60), # `(lower_left_lon, lower_left_lat, upper_right_lon, upper_right_lat)`
        temporal=("2006-01", "2007-01"),

        ## NOTE:
        # lat_range=[49, 60], 
        # long_range=[-120, -110], 
        # grid_resolution=0.5
    )

print(type(ab_terrain))
print("length of ab_terrain: ", len(ab_terrain))

# ned_pipeline.earthdata_save_to_h5(ab_terrain, "scripts/data_collection/static_datasets")
# param_list = ['d_lat', 'd_lon', 'd_UTCTime_40']
dataset = ned.earthdata_slice(h5_file="scripts/data_collection/static_datasets/GLAH06_634_2115_001_1284_4_01_0001.H5",
                                    csv=True, # Save as CSV is turned on
                                    output_dir=static_data_path
                                    )

if os.path.exists("scripts/data_collection/static_datasets/earthdata.csv"):
    logger.info("earthdata.csv is available for inspection")


# import earthaccess
# import pandas as pd
# import h5py
# import logging
# import xarray as xr
# import requests
# import os
# import sys
# from nasa_earthdata_pipeline import NasaEarthdataPipeline as ned

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# static_data_path = "scripts/data_collection/static_datasets"
# ned = ned() # Initialize the NASA Earthdata pipeline (uses login credentials from credentials.json)

# ab_terrain = ned.earthdata_pull_invar(
#         short_name='GLAH06',
#         doi='10.5067/ICESAT/GLAS/DATA109',
#         daac='NSIDC',
#         bounding_box=(-120, 49, -110, 60), # `(lower_left_lon, lower_left_lat, upper_right_lon, upper_right_lat)`
#         temporal=("2006-01", "2007-01"),

#         ## NOTE:
#         # lat_range=[49, 60], 
#         # long_range=[-120, -110], 
#         # grid_resolution=0.5
#     )

# print(type(ab_terrain))
# print("length of ab_terrain: ", len(ab_terrain))

# # ned_pipeline.earthdata_save_to_h5(ab_terrain, "scripts/data_collection/static_datasets")
# # param_list = ['d_lat', 'd_lon', 'd_UTCTime_40']
# dataset = ned.earthdata_slice(h5_file="scripts/data_collection/static_datasets/GLAH06_634_2115_001_1284_4_01_0001.H5",
#                                     csv=True, # Save as CSV is turned on
#                                     output_dir=static_data_path
#                                     )

# if os.path.exists("scripts/data_collection/static_datasets/earthdata.csv"):
#     logger.info("earthdata.csv is available for inspection")



# import earthaccess
# import pandas as pd
# import h5py
# import logging
# import xarray as xr
# import requests
# import os
# from scripts.data_collection.earthdata_pipeline.nasa_earthdata_pipeline import NasaEarthdataPipeline as ned


# # Configure logging
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)

# static_data_path = "scripts/data_collection/static_datasets"

# ned_pipeline = ned(username='jromero7', password='InnoSolve@UW7') # Initialize the pipeline
# ab_terrain = ned_pipeline.earthdata_pull_invar(
#         short_name='GLAH06',
#         doi='10.5067/ICESAT/GLAS/DATA109',
#         daac='NSIDC',
#         bounding_box=(-120, 49, -110, 60), # `(lower_left_lon, lower_left_lat, upper_right_lon, upper_right_lat)`
#         temporal=("2006-01", "2007-01"),
#     )

# print(type(ab_terrain))
# print("length of ab_terrain: ", len(ab_terrain))

# # ned_pipeline.earthdata_save_to_h5(ab_terrain, "scripts/data_collection/static_datasets")
# # param_list = ['d_lat', 'd_lon', 'd_UTCTime_40']
# dataset = ned_pipeline.earthdata_slice(h5_file="scripts/data_collection/static_datasets/GLAH06_634_2115_001_1284_4_01_0001.H5",
#                                     csv=True,
#                                     output_dir=static_data_path
#                                     )

# logger.debug(f"min_lat: {dataset['Latitude'].min()}")
# logger.debug(f"max_lat: {dataset['Latitude'].max()}")
# logger.debug(f"min_lon: {dataset['Longitude'].min()}")
# logger.debug(f"max_lon: {dataset['Longitude'].max()}")

# # ned_pipeline.earthdata_save_to_h5(ab_terrain, static_data_path)
#         # lat_range=[49, 60], 
#         # long_range=[-120, -110], 
#         # grid_resolution=0.5



# import earthaccess
# import pandas as pd
# import h5py
# import logging
# import xarray as xr
# import requests
# import os
# import sys
# from nasa_earthdata_pipeline import NasaEarthdataPipeline as ned

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# static_data_path = "scripts/data_collection/static_datasets"
# ned = ned() # Initialize the NASA Earthdata pipeline (uses login credentials from credentials.json)

# ab_terrain = ned.earthdata_pull_invar(
#         short_name='GLAH06',
#         doi='10.5067/ICESAT/GLAS/DATA109',
#         daac='NSIDC',
#         bounding_box=(-120, 49, -110, 60), # `(lower_left_lon, lower_left_lat, upper_right_lon, upper_right_lat)`
#         temporal=("2006-01", "2007-01"),

#         ## NOTE:
#         # lat_range=[49, 60], 
#         # long_range=[-120, -110], 
#         # grid_resolution=0.5
#     )

# print(type(ab_terrain))
# print("length of ab_terrain: ", len(ab_terrain))

# # ned_pipeline.earthdata_save_to_h5(ab_terrain, "scripts/data_collection/static_datasets")
# # param_list = ['d_lat', 'd_lon', 'd_UTCTime_40']
# dataset = ned.earthdata_slice(h5_file="scripts/data_collection/static_datasets/GLAH06_634_2115_001_1284_4_01_0001.H5",
#                                     csv=True, # Save as CSV is turned on
#                                     output_dir=static_data_path
#                                     )

# if os.path.exists("scripts/data_collection/static_datasets/earthdata.csv"):
#     logger.info("earthdata.csv is available for inspection")



# # import earthaccess
# # import pandas as pd
# # import h5py
# # import logging
# # import xarray as xr
# # import requests
# # import os
# # from scripts.data_collection.earthdata_pipeline.nasa_earthdata_pipeline import NasaEarthdataPipeline as ned


# # # Configure logging
# # logging.basicConfig(level=logging.DEBUG)
# # logger = logging.getLogger(__name__)

# # static_data_path = "scripts/data_collection/static_datasets"

# # ned_pipeline = ned(username='jromero7', password='InnoSolve@UW7') # Initialize the pipeline
# # ab_terrain = ned_pipeline.earthdata_pull_invar(
# #         short_name='GLAH06',
# #         doi='10.5067/ICESAT/GLAS/DATA109',
# #         daac='NSIDC',
# #         bounding_box=(-120, 49, -110, 60), # `(lower_left_lon, lower_left_lat, upper_right_lon, upper_right_lat)`
# #         temporal=("2006-01", "2007-01"),
# #     )

# # print(type(ab_terrain))
# # print("length of ab_terrain: ", len(ab_terrain))

# # # ned_pipeline.earthdata_save_to_h5(ab_terrain, "scripts/data_collection/static_datasets")
# # # param_list = ['d_lat', 'd_lon', 'd_UTCTime_40']
# # dataset = ned_pipeline.earthdata_slice(h5_file="scripts/data_collection/static_datasets/GLAH06_634_2115_001_1284_4_01_0001.H5",
# #                                     csv=True,
# #                                     output_dir=static_data_path
# #                                     )

# # logger.debug(f"min_lat: {dataset['Latitude'].min()}")
# # logger.debug(f"max_lat: {dataset['Latitude'].max()}")
# # logger.debug(f"min_lon: {dataset['Longitude'].min()}")
# # logger.debug(f"max_lon: {dataset['Longitude'].max()}")

# # # ned_pipeline.earthdata_save_to_h5(ab_terrain, static_data_path)
# #         # lat_range=[49, 60], 
# #         # long_range=[-120, -110], 
# #         # grid_resolution=0.5












