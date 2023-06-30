# imports ...
import pandas as pd
import numpy as np
import os
from datetime import datetime, timezone
from sklearn.impute import KNNImputer

# custom functions
from apple_watch_data_extraction import *
from cgm_data_extraction import *
from diary_data_extraction import *
from joining_data_sources import *
from apple_watch_cgm_data_imputation import *


#start_date = '2023-05-26 14:00:00'
path = 'C:\\Users\\y_mus\\OneDrive - Hochschule Luzern\\HSLU\\Thesis\\Patient_raw_data\\Participant_5'

#print(os.listdir(path))
os.chdir(path)

##################################################################################################################################
###################################  Data extraction and merging data sources  ###################################################
##################################################################################################################################

# Extracting metrics from file
df1 = pd.read_csv('AppleWatchData/LUZ05_jun15_jun20.csv') #weekly data 1 
df2 = pd.read_csv('AppleWatchData/LUZ05_jun20_jun25.csv') #weekly data 2

#putting all the apple watch data from the patient together 
df_full = pd.concat([df1, df2], axis=0)
df_full['timestamp'] = pd.to_datetime(df_full['timestamp'])
df_full.set_index('timestamp', inplace=True)

# filling the gaps in the timestamp column
df_full_no_gaps = fill_timestamp_gaps(df_full)

# resampling 
# heart rate, heart rate variability, respiratory rate and blood oxygen saturation --> mean
#df = df_full_no_gaps.resample(rule='5T', offset='2T', label='right',).mean()

# active energy and step count --> sum
#df['active_energy_sum'] = df_full_no_gaps['active_energy'].resample(rule='5T', offset='2T', label='right').sum()
#df['step_count_sum'] = df_full_no_gaps['step_count'].resample(rule='5T', offset='2T', label='right',).sum()


#print(df_full_no_gaps.shape)
##################################################################################################################################

# loading cgm data
cgm_df = cgm_preprocess('LUZ 05 Dexcom Daten.csv')
##################################################################################################################################

# loading manual diary data 
#diary_df = preprocess_diary('LUZ01_Protokoll.xlsx')
##################################################################################################################################

# joining data sources
#df_joined = join_data_sources(df, cgm_df)
df_joined = join_data_sources(df_full_no_gaps, cgm_df)
#df_joined = join_data_sources(df_full_no_gaps, cgm_df, diary_df)
##################################################################################################################################


##################################################################################################################################

# discarding everything before the start time (time when participant put on the watch)
df_joined.set_index('timestamp', inplace=True)
#df_joined = pd.DataFrame(df_joined[df_joined.index > start_date])

# saving the results
#df_joined.to_csv('P1_watch_cgm_diary.csv')
df_joined.to_csv('LUZ05_watch_cgm_data.csv')

#print(df_joined.head(10))  
print('joined data saved sucessfully')


