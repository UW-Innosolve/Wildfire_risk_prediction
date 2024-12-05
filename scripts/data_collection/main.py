import alberta_wf_incidence_loader
import cds_pipeline
import raw_data_assembly

def main():

    ## WILDFIRE INCIDENCE DATA
    wildfire_data_path = "scripts/data_collection/fp-historical-wildfire-data-2006-2023.xlsx"

    ## Load wildfire incidence data
    wildfire_loader = alberta_wf_incidence_loader.AlbertaWildfireIncidenceLoader(wildfire_data_path)

    ## Resample wildfire incidence data
    start_date = "2006-01-01"
    end_date = "2023-12-31"
    interval = '4D'
    resampled_fire_incidents = wildfire_loader.wildfire_incidence_data_resample(start_date, end_date, interval, wildfire_loader.ab_fire_incidents)
    wildfire_incidence_data = resampled_fire_incidents


    ## CDS PIPELINE
    ## Initialize cds pipeline
    cds_pipeline = cds_pipeline.CdsPipeline(key='734d2638-ef39-4dc1-bc54-4842b788fff6')
    ## Set CDS time-variant variables
    cds_pipeline.set_variant_variables(['2m_temperature', 
                                        'surface_pressure', 
                                        '10m_u_component_of_wind', 
                                        '10m_v_component_of_wind', 
                                        '2m_dewpoint_temperature', 
                                        'total_precipitation'])
    
    ## Set CDS time-invariant variables # NOTE: Above variables are placeholders, actual variables will be added later
    ## NOTE: api behaviour for invariant variables is not yet tested.
    cds_pipeline.set_invariant_variables(['slt', 'tvl', 'tvh']) # these are soil type, type low vegetation, and type high vegetation

    ## Set CDS request parameters
    cds_pipeline.set_request_parameters(cds_pipeline.variables, cds_pipeline.invariant_variables, start_date, end_date, [49, 60], [-120, -110], 0.5)


    ## RAW DATA ASSEMBLY
    ## Create pipelines list
    pipelines = [{'CDS': cds_pipeline}]

    ## Initialize the raw data assembly
    raw_data_assembly = raw_data_assembly.RawDataAssembler(wildfire_incidence_data)

    ## Assemble the dataset
    raw_data_assembly.assemble_dataset(pipelines, 'M', latitude_tolerance=0.1, longitude_tolerance=0.1)