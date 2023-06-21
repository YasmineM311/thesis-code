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


start_date = '2023-05-11 11:14:00'
path = 'C:\\Users\\y_mus\\OneDrive - Hochschule Luzern\\HSLU\\Thesis\\Patient_raw_data\\Participant_1'

#print(os.listdir(path))
os.chdir(path)

##################################################################################################################################
###################################  Data extraction and merging data sources  ###################################################
##################################################################################################################################

# Extracting metrics from file
df1 = fetch_data_from_file('AppleWatchData/WeeklyData/LUZ01_11_17_may.json') #weekly data 1
df2 = fetch_data_from_file('AppleWatchData/WeeklyData/LUZ01_may_13_19.json') #weekly data 2
df3 = fetch_data_from_file('AppleWatchData/LUZ01_lastday_may20.json') #last day data

# subsetting the first two days only, the rest of the data is present in the second file
df1 = df1[df1.index < '2023-05-13'] 

# subsetting the last day only
df3 = df3[df3.index > '2023-05-20']

#putting all the apple watch data from the patient together 
df_full = pd.concat([df1, df2, df3], axis=0)

# filling the gaps in the timestamp column
df_full_no_gaps = fill_timestamp_gaps(df_full)

#print(df_full_no_gaps.shape)
##################################################################################################################################

# loading cgm data
cgm_df = cgm_preprocess('LUZ 01 Dexcom Daten.csv')
##################################################################################################################################

# loading manual diary data 
diary_df = preprocess_diary('LUZ01_Protokoll.xlsx')
##################################################################################################################################

# joining data sources
df_joined = join_data_sources(df_full_no_gaps, cgm_df, diary_df)
##################################################################################################################################

# specifying sleep timeframes to create a binary 'sleep' column
df_joined['sleep'] = np.where(((df_joined['timestamp'] > '2023-05-11 23:45:00') & (df_joined['timestamp'] < '2023-05-12 06:20:00'))
                               |
                               ((df_joined['timestamp'] > '2023-05-12 23:45:00') & (df_joined['timestamp'] < '2023-05-13 08:45:00'))
                               |
                               ((df_joined['timestamp'] > '2023-05-14 01:00:00') & (df_joined['timestamp'] < '2023-05-14 08:30:00'))
                               |
                               ((df_joined['timestamp'] > '2023-05-14 23:45:00') & (df_joined['timestamp'] < '2023-05-15 06:15:00'))
                               |
                               ((df_joined['timestamp'] > '2023-05-15 23:30:00') & (df_joined['timestamp'] < '2023-05-16 06:15:00'))
                               |
                               ((df_joined['timestamp'] > '2023-05-17 00:00:00') & (df_joined['timestamp'] < '2023-05-17 06:15:00'))
                               |
                               ((df_joined['timestamp'] > '2023-05-18 00:45:00') & (df_joined['timestamp'] < '2023-05-18 08:30:00'))
                               |
                               ((df_joined['timestamp'] > '2023-05-19 00:45:00') & (df_joined['timestamp'] < '2023-05-19 09:20:00'))
                               |
                               ((df_joined['timestamp'] > '2023-05-19 23:00:00') & (df_joined['timestamp'] < '2023-05-20 06:15:00')), 1, 0)
##################################################################################################################################

# discarding everything before the start time (time when participant put on the watch)
df_joined.set_index('timestamp', inplace=True)
df_joined = pd.DataFrame(df_joined[df_joined.index > start_date])

# saving the results
df_joined.to_csv('P1_watch_cgm_diary.csv')

#print(df_joined.head(10))  
print('joined data saved sucessfully')

##################################################################################################################################
###############################################  AW and CGM data imputation  #####################################################
##################################################################################################################################

# imputing apple watch data
#df_imputed1 = impute_applewatch_data(df_joined)

# imputing cgm data
#df_imputed2 = cgm_data_backward_fill(df_imputed1)

#df_imputed2.to_csv('P1_imputed_data.csv')

#print('imputed data saved sucessfully')
##################################################################################################################################

