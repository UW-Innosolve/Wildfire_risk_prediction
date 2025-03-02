import os
from ab_lightning_pipeline.ab_lightning_pipeline import AbLightningPipeline

data_dir = "scripts/data_collection/static_datasets/ablightning_historical"

ablp = AbLightningPipeline(data_dir)

ldf = ablp.get_ltng_df()
print(ldf.head())

