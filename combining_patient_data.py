import os
import pandas as pd
from health_data_functions import *

## make sure you are pointing to the correct directory and that only has json files ##

# creating a list of all files in the directory
file_list = os.listdir()

# creating a dictionary to store the dataframe corresponding to each file 
dataframe_dict = {}

for i, file in enumerate(file_list):
    dataframe_dict[i] = fetch_data_from_file(file)

# converting the dict values (ie dataframes) into a list and then concatenating them 
# into one dataframe
dataframe_list = list(dataframe_dict.values())
full_df = pd.concat(dataframe_list)

full_df.to_csv('full_df1.csv')

print(full_df.shape)


