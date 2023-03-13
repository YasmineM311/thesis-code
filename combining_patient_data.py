import os
import pandas as pd
from health_data_functions import *

## make sure you are pointing to the correct directory

# creating a list of all files in the directory
file_list = os.listdir()

# creating a list of names to be used as dictionary keys 
# each corresponding to one of the files in the directory
filenames = []
for i in range(1,len(file_list)+1):
    name = "file_" + str(i)
    filenames.append(name)

# creating a dictionary to store the dataframe corresponding to each file 
dataframe_dict = {}

for i in range(len(file_list)):
    name = filenames[i]
    dataframe_dict[name] = fetch_data_from_file(file_list[i])

dataframe_list = []
for df in dataframe_dict.values():
    dataframe_list.append(df)

full_df = pd.concat(dataframe_list)
full_df.to_csv('full_df.csv')

print('done')

