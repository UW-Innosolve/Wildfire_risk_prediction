# main.py

import time
import logging

# Import Pipeline classes
from earthkit_pipeline.earthkit_pipeline import EkPipeline
from oapi_pipeline.human_activity_pipeline import HumanActivityPipeline
from ablightning_pipeline.ab_lightning_pipeline import AbLightningPipeline
# from earthdata_pipeline.nasa_earthdata_pipeline import NasaEarthdataPipeline as ned

# Import Utility classes
import collection_utils.alberta_wf_incidence_loader as alberta_wf_incidence_loader
import collection_utils.raw_data_assembly as raw_data_assembly

from earthkit_pipeline.cds_auth import CdsAuth
# import cfgrib
# import eccodes

# Configure logging: Log to both file (pipeline.log) and the console.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(name)s: %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    # ------------------------------------------------------
    # 1. Load Wildfire Incidence Data
    # ------------------------------------------------------
    wildfire_data_path = "scripts/data_collection/static_datasets/fp-historical-wildfire-data-2006-2023.xlsx"
    wildfire_loader = alberta_wf_incidence_loader.AlbertaWildfireIncidenceLoader(wildfire_data_path)
    wildfire_incidence_data = wildfire_loader.ab_fire_incidents
    # Log the number of incidents loaded.
    logger.debug(f"Wildfire Incidence Data Columns in main: {wildfire_incidence_data.columns}")

    # ------------------------------------------------------
    # 2. Define Shared Query Parameters (for all pipelines)
    # ------------------------------------------------------
    # Define the geographic bounds (latitude and longitude ranges)
    query_area = {
        'latitude_range': [49, 60],
        'longitude_range': [-120, -110]
    }
    # Define the temporal period for data retrieval
    query_period = {
        'start_date': '2006-01-01',
        'end_date': '2024-12-31'
    }
    # Define the grid resolution (in degrees)
    query_grid_resolution = 0.30

    # ------------------------------------------------------
    # 3. Pipeline Configuration: Enable/Disable Pipelines
    # ------------------------------------------------------
    # Set each flag to True or False to control which pipeline is initialized.
    pipeline_config = {
        "EARTHKIT": True,         # For CDS ERA5 reanalysis data
        "AB_LIGHTNING": True,      # For Alberta Lightning data
        "HUMAN_ACTIVITY": True,    # For Human Activity data (OSM-based)
        # "NED": False,           # Example for NASA Earthdata pipeline (if available)
    }

    # We'll build a list of pipeline dictionaries. Each dictionary will have one key representing the pipeline type.
    pipelines = []

    # ------------------------------------------------------
    # 4. Initialize the EARTHKIT Pipeline (CDS ERA5 Data)
    # ------------------------------------------------------
    if pipeline_config["EARTHKIT"]:
        # Retrieve the CDS API key from credentials.json via the CdsAuth class.
        cds_key = CdsAuth().get_cds_key(cred_file_path="scripts/data_collection/credentials.JSON")
        ek_pipeline = EkPipeline(cds_key)
        # Define the variables for the CDS request:
        # - Variant parameters (time-dependent)
        variant_cds_params = [
            '2t',      # 2m_temperature
            '10u',     # 10m_u_component_of_wind
            '10v',     # 10m_v_component_of_wind
            '2d',      # 2m_dewpoint_temperature
            'lai_lv',  # Leaf area index for low vegetation
            'lai_hv',  # Leaf area index for high vegetation
            'swvl1', 'swvl2', 'swvl3', 'swvl4',  # Soil water layers
            'stl1', 'stl2', 'stl3', 'stl4',       # Soil temperature levels
            'sp',      # Surface pressure
            'fal',     # Forest albedo
            'src'      # Skin reservoir content
        ]
        # - Invariant parameters (time-independent)
        invariant_cds_params = [
            'tvl', 'tvh',  # Vegetation cover (low and high)
            'cvl', 'cvh',  # Vegetation type (low and high)
            'cl',          # Lake cover
            'lsm',         # Land-sea mask
            'z',           # Geopotential (proportional to elevation)
            'slt'          # Soil type
        ]
        # - Accumulated parameters (e.g., precipitation)
        accumulated_cds_params = [
            'tp',   # Total precipitation
            'e',    # Evaporation
            'pev',  # Potential evaporation
            'sshf', # Surface sensible heat flux
            'slhf', # Surface latent heat flux
            'ssrd', # Surface solar radiation downwards
            'strd', # Surface thermal radiation downwards
            'ssr',  # Surface net solar radiation
            'str'   # Surface net thermal radiation
        ]
        # Set the request parameters in Earthkit.
        ek_pipeline.set_cds_request_parameters(
            var_params=variant_cds_params,
            invar_params=invariant_cds_params,
            accum_params=accumulated_cds_params,
            lat_range=query_area['latitude_range'],
            long_range=query_area['longitude_range'],
            grid_resolution=query_grid_resolution
        )
        # Add this pipeline to our list with the key "EARTHKIT".
        pipelines.append({"EARTHKIT": ek_pipeline})

    # ------------------------------------------------------
    # 5. Initialize the AB LIGHTNING Pipeline
    # ------------------------------------------------------
    if pipeline_config["AB_LIGHTNING"]:
        abltng = AbLightningPipeline("scripts/data_collection/static_datasets/ablightning_historical")
        abltng.set_ab_ltng_params(
            lat_range=query_area['latitude_range'],
            lon_range=query_area['longitude_range'],
            grid_resolution=query_grid_resolution
        )
        pipelines.append({"AB_LIGHTNING": abltng})

    # ------------------------------------------------------
    # 6. Initialize the HUMAN ACTIVITY Pipeline
    # ------------------------------------------------------
    if pipeline_config["HUMAN_ACTIVITY"]:
        oapi_pipeline = HumanActivityPipeline()
        oapi_pipeline.set_osm_params(
            lat_range=query_area['latitude_range'],
            lon_range=query_area['longitude_range'],
            grid_resolution=query_grid_resolution
        )
        pipelines.append({"HUMAN_ACTIVITY": oapi_pipeline})

    # (Optional) Initialize additional pipelines here, e.g., NASA Earthdata.
    # if pipeline_config["NED"]:
    #     ned_pipeline = NasaEarthdataPipeline(...)
    #     pipelines.append({"NED": ned_pipeline})

    logger.info(f"Pipeline list: {pipelines}")

    # ------------------------------------------------------
    # 7. Raw Data Assembly: Integrate Data from Active Pipelines
    # ------------------------------------------------------
    raw_data_assembler = raw_data_assembly.RawDataAssembler(
        wildfire_incidence_data,
        start_date=query_period['start_date'],
        end_date=query_period['end_date'],
        resample_interval='1D',
        grouping_period_size='M',
        latitude_tolerance=1.0,
        longitude_tolerance=1.0
    )

    # The assembler will loop over each grouped period (e.g., monthly)
    # and merge the outputs from each active pipeline.
    raw_data_assembler.assemble_dataset(pipelines)
    # Note: The assembled data is saved as monthly CSV files in an output folder.

if __name__ == "__main__":
    main()
