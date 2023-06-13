# imports ...
import pandas as pd
import numpy as np
import os
from datetime import datetime, timezone
from sklearn.impute import KNNImputer
from pyod.models.ecod import ECOD


# custom functions
from apple_watch_data_extraction import *
from cgm_data_extraction import *
from diary_data_extraction import *
from joining_data_sources import *
from apple_watch_cgm_data_imputation import *


path = 'C:\\Users\\y_mus\\OneDrive - Hochschule Luzern\\HSLU\\Thesis\\Patient_raw_data\\Participant_1'

#print(os.listdir(path))
os.chdir(path)

##################################################################################################################################
###############################################  AW and CGM data imputation  #####################################################
##################################################################################################################################

# load merged data
df_joined = pd.read_csv('P1_watch_cgm_diary_sleep_corrected.csv', index_col='timestamp') 

# imputing apple watch data
df_imputed1 = impute_applewatch_data(df_joined)

# imputing cgm data
df_imputed2 = cgm_data_backward_fill(df_imputed1)

df_imputed2.to_csv('P1_imputed_data.csv')

print('imputed data saved sucessfully')
##################################################################################################################################

