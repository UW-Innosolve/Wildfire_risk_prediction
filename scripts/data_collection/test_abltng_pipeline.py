import os
from ablightning_pipeline.ab_lightning_pipeline import AbLightningPipeline


abltng = AbLightningPipeline("scripts/data_collection/static_datasets/ablightning_historical")
raw_ltng = abltng.get_raw_ltng_df()
print(raw_ltng.head())
abltng.set_ab_ltng_params(
    lat_range=[49, 60],
    lon_range=[-120, -110],
    grid_resolution=0.3
)

batch_dates = ['2019-01-01', '2019-01-02', '2019-01-03', '2019-01-04', '2019-01-05']

final_gridded_and_dated_ltng_data = abltng.get_ltng_data(batch_dates)
print(final_gridded_and_dated_ltng_data.head())


