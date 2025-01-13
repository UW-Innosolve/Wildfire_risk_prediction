import earthaccess
import pandas as pd
import h5py
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# # Set up the Earthdata login credentials
# EARTHDATA_USERNAME = 'jromero7'
# EARTHDATA_PASSWORD = 'InnoSolve@UW7'

# Initialize the Earthdata login
earthaccess.login(strategy='netrc',)


# Sample search for the ATL06 dataset
logger.info("Searching for the GLAH06 dataset in NSIDC DAAC")
results = earthaccess.search_data(
    short_name='GLAH06', # GLAS/ICESat L2 Global Land Surface Altimetry Data
    daac='NSIDC', # National Snow and Ice Data Center (NSIDC) DAAC
    bounding_box=(-10, 20, 10, 50),
    temporal=("2006-01", "2006-02"),
    count=1,
)

# Download the data
logger.info("Downloading the GLAH06 dataset")
earthaccess.download(results, "scripts/data_collection/earthaccess_samples")


# Open the HDF5 file
for granule in results:
    logger.info(f"Opening the HDF5 file: {file}")
    # file_path = granule.data_links[0]
    with h5py.File(file, 'r') as f:
        # Assume the dataset is stored under the key 'dataset_name'
        data = f['dataset_name'][:]
        # Convert to DataFrame
        df = pd.DataFrame(data)


# Display the DataFrame
print(df)

logger.info("Saving the GLAH06 dataset as a CSV file")
df.to_csv("scripts/data_collection/earthaccess_samples/GLAH06_data_sample.csv")






files = earthaccess.download(results, "scripts/data_collection/earthaccess_samples")

1