# imports 
import pandas as pd
from health_data_functions import *


df = fetch_data_from_file('C:/Users/y_mus/OneDrive - Hochschule Luzern/HSLU/Thesis/Code/my apple watch data/HealthAutoExport-2023-01-25.json')
df.to_csv('test.csv')


