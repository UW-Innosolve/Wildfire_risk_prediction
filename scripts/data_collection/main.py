# main.py

import time

## Import Pipeline classes
from earthkit_pipeline.earthkit_pipeline import EkPipeline
from oapi_pipeline.human_activity_pipeline import HumanActivityPipeline
from ablightning_pipeline.ab_lightning_pipeline import AbLightningPipeline
# from earthdata_pipeline.nasa_earthdata_pipeline import NasaEarthdataPipeline as ned

# Import Utility classes
import collection_utils.alberta_wf_incidence_loader as alberta_wf_incidence_loader
import collection_utils.raw_data_assembly as raw_data_assembly

import logging
from earthkit_pipeline.cds_auth import CdsAuth
import cfgrib
import eccodes

# Configure logging
logging.basicConfig(
    level=logging.info,
    format='%(asctime)s %(levelname)s:%(name)s: %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    ## WILDFIRE INCIDENCE DATA
    wildfire_data_path = "scripts/data_collection/static_datasets/fp-historical-wildfire-data-2006-2023.xlsx"

    ## Load wildfire incidence data
    wildfire_loader = alberta_wf_incidence_loader.AlbertaWildfireIncidenceLoader(wildfire_data_path)

    ## wildfire_incidence_data
    wildfire_incidence_data = wildfire_loader.ab_fire_incidents
    
    ## Set query parameters (to be used by all pipelines)
    query_area = {'latitude_range': [49, 60], 'longitude_range': [-120, -110]}
    query_period = {'start_date': '2014-01-01', 'end_date': '2014-01-15'}
    query_grid_resolution = 0.35

    # Debug: Print columns to verify 'fire_start_date' exists
    logger.debug(f"Wildfire Incidence Data Columns in main: {wildfire_incidence_data.columns}")

    ## EARTHKIT PIPELINE ############################################################
    ## Initialize CDS pipeline
    cds_key = CdsAuth().get_cds_key(cred_file_path="scripts/data_collection/credentials.JSON") # Get CDS API key from credentials file
    ek_pipeline = EkPipeline(cds_key)
    
    ## Set CDS era5 pipeline parameters
    variant_cds_params = [  # Variant variables
                            # Temperature and pressure
                            '2t',      # 2m_temperature
                            # Wind
                            '10u',      # 10m_u_component_of_wind', 
                            '10v',      # 10m_v_component_of_wind',
                            # # Water variables
                            '2d',      # 2m_dewpoint_temperature',                             
                            # Leaf area index (vegetation)
                            'lai_lv',   # leaf_area_index_low_vegetation',
                            'lai_hv'   # leaf_area_index_high_vegetation',
                            # Soil Water Layer
                            'swvl1', # volumetric_soil_water_layer_1',
                            'swvl2', # volumetric_soil_water_layer_2',
                            'swvl3', # volumetric_soil_water_layer_3',
                            'swvl4', # volumetric_soil_water_layer_4',
                            # Soil temperature
                            'stl1', # soil_temperature_level_1',
                            'stl2', # soil_temperature_level_2',
                            'stl3', # soil_temperature_level_3',
                            'stl4', # soil_temperature_level_4',
                            # Pressure levels
                            'sp', # surface_pressure',
                            # Forest albedo
                            'fal', # forest_albedo',
                            # Skin reservoir content
                            'src', # skin_reservoir_content',
                            
    ]    
    
    invariant_cds_params = [ # Invariant variables
                            # Vegetation cover and type
                            'tvl', # low_veg_cover
                            'tvh', # high_veg_cover
                            'cvl', # low_veg_type
                            'cvh',  # high_veg_type
                             # Lakes and rivers
                            'cl',  # lake_cover
                            'lsm', # land_sea_mask
                             # Topography
                            'z',    # Geopotential (proportional to elevation, not linearly due to oblong shape of Earth)
                            'slt' # Soil type
    ]
    
    accumulated_cds_params = [  # Accumulated variables
                                'tp',       # total_precipitation',
                                # Evaopration variables
                                'e',        # evaporation',
                                'pev',      # potential_evaporation',
                                # Heat and radiation variables
                                'sshf',      # surface_sensible_heat_flux',
                                'slhf',      # surface_latent_heat_flux',
                                'ssrd',      # surface_solar_radiation_downwards',
                                'strd',      # surface_thermal_radiation_downwards',
                                'ssr',       # surface_net_solar_radiation
                                'str',       # surface_net_thermal_radiation
    ]


    ## Set CDS request parameters
    ek_pipeline.set_cds_request_parameters(
        var_params=variant_cds_params, 
        invar_params=invariant_cds_params, 
        accum_params=accumulated_cds_params,
        lat_range=query_area['latitude_range'],
        long_range=query_area['longitude_range'],
        grid_resolution=query_grid_resolution
    )
    
    ## AB LIGHTNING PIPELINE ############################################################
    
    abltng = AbLightningPipeline("scripts/data_collection/static_datasets/ablightning_historical")
    abltng.set_ab_ltng_params(
        lat_range=query_area['latitude_range'],
        lon_range=query_area['longitude_range'],
        grid_resolution=query_grid_resolution
    )
    
    ## HUMAN ACTIVITY PIPELINE ############################################################

    # Create OAPI pipeline object
    # oapi_pipeline = HumanActivityPipeline() 
    # NOTE: Creating the pipeline object, before passing it to the pipeline list
    #       is prefered to avoid the pipeline object being overwritten by the next pipeline object.
    #       However, human activity pipeline currently relies on earthkit data existing already.
    # TODO: Correct human activity pipeline to be able to run independently of earthkit data.
    
    ## NASA EARTHDATA PIPELINE ############################################################
    
    
    
    ########################################################################################
    
    ## RAW DATA ASSEMBLY
    ## Pipeline object list
    pipelines = [
        {'EARTHKIT': ek_pipeline},
        {'AB_LIGHTNING': abltng},
        # {'HUMAN_ACTIVITY': HumanActivityPipeline()}
    ]

    ## Initialize the raw data assembler
    raw_data_assembly_instance = raw_data_assembly.RawDataAssembler(
        wildfire_incidence_data, 
        start_date=query_period['start_date'], 
        end_date=query_period['end_date'], 
        resample_interval='4D',
        grouping_period_size='M',
        latitude_tolerance=1.0,
        longitude_tolerance=1.0
    )

    ## Assemble the dataset
    raw_data_assembly_instance.assemble_dataset(pipelines)
    ## Note raw data assembly saves monthly csv data to folder in root directory


if __name__ == "__main__":
    main()
