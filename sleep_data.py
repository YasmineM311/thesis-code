import json
import pandas as pd
import os
from apple_watch_data_extraction import *

## make sure you are in the correct directory

file_list = os.listdir()

# initiating an empty dict and populating it with extracted sleep data
sleep_data_dict = {}

for i, file in enumerate(file_list):
    df = pd.DataFrame(transform(pd.read_json(file)))
    sleep = df[df.name=='sleep_analysis'][['date', 'sleepStart', 'sleepEnd', 'rem', 'core', 'deep']]
    sleep_data_dict[i] = sleep

# converting the dict into a list to concatenate its components into a dataframe
sleep_data_list = list(sleep_data_dict.values())
df = pd.concat(sleep_data_list)

df.to_csv('patient_xyz_sleep.csv')
print(df)