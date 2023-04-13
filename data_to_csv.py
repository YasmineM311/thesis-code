# imports 
import pandas as pd
from health_data_functions import *

pd.options.display.max_rows = 2000

## make sure you are in the correct directory

# Extracting metrics from file
df = fetch_data_from_file('HealthAutoExport-2023-03-07.json')
# df.to_csv('test.csv')

print(df.head())




