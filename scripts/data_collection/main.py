# main.py

import alberta_wf_incidence_loader
from CDS_pipeline import CdsPipeline
import raw_data_assembly
import logging

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

def main():

    ## WILDFIRE INCIDENCE DATA
    wildfire_data_path = "scripts/data_collection/fp-historical-wildfire-data-2006-2023.xlsx"

    ## Load wildfire incidence data
    wildfire_loader = alberta_wf_incidence_loader.AlbertaWildfireIncidenceLoader(wildfire_data_path)

    ## Resample wildfire incidence data
    start_date = "2006-01-01"
    end_date = "2023-12-31"
    interval = '4D'
    resampled_fire_incidents = wildfire_loader.wildfire_incidence_data_resample(
        start_date, end_date, interval, wildfire_loader.ab_fire_incidents
    )
    wildfire_incidence_data = resampled_fire_incidents

    # Debug: Print columns to verify 'fire_start_date' exists
    logger.info(f"Wildfire Incidence Data Columns in main: {wildfire_incidence_data.columns}")

    ## CDS PIPELINE
    ## Initialize CDS pipeline
    cds_pipeline = CdsPipeline(key='734d2638-ef39-4dc1-bc54-4842b788fff6')
    
    ## Set CDS time-variant variables
    cds_pipeline.set_variant_variables([
        '2m_temperature', 
        'surface_pressure', 
        '10m_u_component_of_wind', 
        '10m_v_component_of_wind', 
        '2m_dewpoint_temperature', 
        'total_precipitation'
    ])
    
    ## Set CDS time-invariant variables 
    ## Using correct abbreviated variable names
    cds_pipeline.set_invariant_variables(['slt', 'tvl', 'tvh'])  # soil type, type low vegetation, and type high vegetation

    ## Set CDS request parameters
    cds_pipeline.set_request_parameters(
        var_variables=cds_pipeline.var_variables, 
        invar_variables=cds_pipeline.invar_variables, 
        lat_range=[49, 60], 
        long_range=[-120, -110], 
        grid_resolution=0.5
    )

    ## RAW DATA ASSEMBLY
    ## Create pipelines list
    pipelines = [{'CDS': cds_pipeline}]

    ## Initialize the raw data assembly
    raw_data_assembly_instance = raw_data_assembly.RawDataAssembler(wildfire_incidence_data)

    ## Assemble the dataset
    raw_data_assembly_instance.assemble_dataset(
        pipelines, 
        grouping_period_size='M', 
        latitude_tolerance=1.0,  # Increased tolerance to 0.5 degree
        longitude_tolerance=1.0
    )

if __name__ == "__main__":
    main()
