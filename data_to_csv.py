# imports 
import pandas as pd
from health_data_functions import *

pd.options.display.max_rows = 2000

df = fetch_data_from_file('C:/Users/y_mus/OneDrive - Hochschule Luzern/HSLU/Thesis/Code/my apple watch data/HealthAutoExport-2023-01-22.json')
df.to_csv('test.csv')

# print(df[df.index > "12:00"])




