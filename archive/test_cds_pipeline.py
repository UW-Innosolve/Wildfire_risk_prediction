from cds_pipeline.CDS_pipeline import CdsPipeline as cds
import data_collection.collection_utils.raw_data_assembly as raw_data_assembly
import data_collection.collection_utils.alberta_wf_incidence_loader as alberta_wf_incidence_loader
# from oapi_pipeline.human_activity_pipeline import HumanActivityPipeline
import logging
import time
from cds_pipeline.cds_auth import CdsAuth


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(name)s: %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# # NEW IMPORT for the Human Activity Pipeline
# from human_activity_pipeline import HumanActivityPipeline

def main():
    ## WILDFIRE INCIDENCE DATA
    wildfire_data_path = "scripts/data_collection/static_datasets/fp-historical-wildfire-data-2006-2023.xlsx"

    ## Load wildfire incidence data
    wildfire_loader = alberta_wf_incidence_loader.AlbertaWildfireIncidenceLoader(wildfire_data_path)

    ## wildfire_incidence_data
    wildfire_incidence_data = wildfire_loader.ab_fire_incidents

    # Debug: Print columns to verify 'fire_start_date' exists
    logger.info(f"Wildfire Incidence Data Columns in main: {wildfire_incidence_data.columns}")

    ## CDS PIPELINE
    ## Initialize CDS pipeline
    cds_key = CdsAuth().get_cds_key(cred_file_path="scripts/data_collection/credentials.JSON") # Get CDS API key from credentials file
    cds_pipeline = CdsPipeline(cds_key)
    
    
    ## Set CDS time-invariant variables (parameters)
    invariant_cds_params = ['low_veg_cover', 
                            'high_veg_cover', 
                            'low_veg_type', 
                            'high_veg_type']
    
    ## Set CDS time-variant variables
    variant_cds_params = [  # Temperature and pressure
                            '2m_temperature', 
                            'surface_pressure',
                            # Wind
                            '10m_u_component_of_wind', 
                            '10m_v_component_of_wind',
                            # Water variables
                            '2m_dewpoint_temperature', 
                            'total_precipitation',
                            'total_evaporation',
                            # Leaf area index (vegetation)
                            'leaf_area_index_low_vegetation',
                            'leaf_area_index_high_vegetation',
                            # Heat variables (NOTE: needs review and/or reduction)
                            'surface_sensible_heat_flux',
                            'surface_latent_heat_flux',
                            'surface_solar_radiation_downwards',
                            'surface_thermal_radiation_downwards',
                            'surface_net_solar_radiation',
                            'surface_net_thermal_radiation',
    ]


    ## Set CDS request parameters
    cds_pipeline.set_request_parameters(
        var_params=variant_cds_params, 
        invar_params=invariant_cds_params, 
        lat_range=[49, 60], 
        long_range=[-120, -110], 
        grid_resolution=0.5
    )
    
    ## RAW DATA ASSEMBLY
    ## Create pipelines list
    pipelines = [
        {'CDS': cds_pipeline},
        # {'HUMAN_ACTIVITY': HumanActivityPipeline()},
    ]

    ## Initialize the raw data assembler
    raw_data_assembly_instance = raw_data_assembly.RawDataAssembler(
        wildfire_incidence_data, 
        start_date='2006-01-01', 
        end_date='2023-12-31', 
        resample_interval='4D',
        grouping_period_size='M',
        latitude_tolerance=1.0,
        longitude_tolerance=1.0
    )
    
    invar_data = cds_pipeline.get_invariant_data()
    

    ## Assemble the dataset
    raw_data_assembly_instance.assemble_dataset(pipelines)


if __name__ == "__main__":
    main()
