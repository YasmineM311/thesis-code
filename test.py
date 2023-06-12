import os
import pandas as pd
from datetime import datetime, timezone
from apple_watch_data_extraction import *


file = "C:\\Users\\y_mus\\OneDrive - Hochschule Luzern\\HSLU\Thesis\\Patient_raw_data\\Yasmine's data\\applewatchdata_24Mar_03Apr.json"

health_data = pd.read_json(file)

data = transform(health_data)

print(data[0:10])