### FOR VISUALIZATION OF RAW DATA
### just to be able to see visual patterns and get initial feel for the data

import numpy as np

import pandas as pd
import matplotlib.pyplot as plt


## import sample raw data file
# pulling July 2014 from local path on Teo's machine
raw_path = "/Users/teodoravujovic/Desktop/data/firebird/march13_pull/fb_raw_data_201407.csv"

# load raw data into pandas
raw_df = pd.read_csv(raw_path)

# columns from df
# columns_used = raw_df.columns[3:]
columns_used = ['10u', '10v', '2d', '2t', 'cl', 'cvh',
       'cvl', 'fal', 'lai_hv', 'lai_lv', 'lsm', 'slt', 'sp', 'src', 'stl1',
       'stl2', 'stl3', 'stl4', 'swvl1', 'swvl2', 'swvl3', 'swvl4', 'tvh',
       'tvl', 'z', 'e', 'pev', 'slhf', 'sshf', 'ssr', 'ssrd', 'str', 'strd',
       'tp', 'is_fire_day', 'lightning_count', 'absv_strength_sum',
       'multiplicity_sum', 'railway_count', 'power_line_count',
       'highway_count', 'aeroway_count', 'waterway_count']


# generate and save an figure for each
for i in range(31):
    day = raw_df[i*1258:(i+1)*1258] # each day has 1258 rows
    for parameter in columns_used:
        values = np.asarray(day[parameter]).reshape(37,34) # each day has 37 lines of latitude and 34 lines of longtitude
        plt.imshow(values, interpolation='nearest') # TODO: do I need the interpolation here?
        # plt.show(fig) # to show instead of save the image
        plt.savefig(f"{parameter}_day{i}.png", bbox_inches='tight')


print('pause')
