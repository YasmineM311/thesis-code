import pandas as pd
import numpy as np
import statistics
from datetime import datetime
from datetime import timedelta


'''
This is a config file containing all functions necessary for feature engineering and imputation.
'''


def create_centred_metric(df, metric_name):
    
    '''
    Returns a scaled or zero-centered metric.
    The scaling/centering is done for each individual day/night to account for day to day and night to night fluctuations. 
    '''
    
    
    # scaled_values = []
    centred_values = []
    
    # grouping df by each day and night
    for group in df.groupby([df.day_night]):
        
        ## standrdization
        #X = group[1]['heart_rate'].values.reshape(-1, 1) # slicing hr values for given group
        #transformer = StandardScaler().fit(X)
        #scaled_values.append(transformer.transform(X))
    
        ## median centering
        X = group[1][metric_name]
        X_sorted = group[1][metric_name].dropna().sort_values(ascending=True) #dropping NaNs and sorting values to calculate the median
        
        if len(X_sorted) > 0:
            median_value = statistics.median(X_sorted)
            centred_values.append(X - median_value)
        else:
            centred_values.append([np.nan] * len(group[1]))
    
    
    #scaled_values = np.concatenate(scaled_values) # convert list of arrays to a single list
    #df[centred_metric_name] = scaled_values

    centred_values = list(np.concatenate(centred_values))
    #df[centred_metric_name] = centred_values
    
    #return df
    return centred_values
    
#############################################################################################################################################    

def hr_rolling_features(df):

    '''
    Returns dataframe with heart rate rolling features.
    '''
    
    ## from centred values
    df['hrcentred_5min_rolling'] = df['hr_centred'].rolling(window=5, min_periods=1).apply(lambda x: x[x!= 0].mean()) # short term feature
    df['hrcentred_10min_rolling'] = df['hr_centred'].rolling(window=10, min_periods=1).apply(lambda x: x[x!= 0].mean()) # short term feature
    #df['hrcentred_15min_rolling'] = df['hr_centred'].rolling(window=15, min_periods=1).apply(lambda x: x[x!= 0].mean()) # short term feature
    df['hrcentred_30min_rolling'] = df['hr_centred'].rolling(window=30, min_periods=1).apply(lambda x: x[x!= 0].mean()) # long term feature
    df['hrcentred_60min_rolling'] = df['hr_centred'].rolling(window=60, min_periods=1).apply(lambda x: x[x!= 0].mean()) # long term feature
    
    ## from original values
    df['hr_5min_rolling'] = df['heart_rate'].rolling(window=5, min_periods=1).apply(lambda x: x[x!= 0].mean()) # short term feature
    df['hr_10min_rolling'] = df['heart_rate'].rolling(window=10, min_periods=1).apply(lambda x: x[x!= 0].mean()) # short term feature
    #df['hr_15min_rolling'] = df['heart_rate'].rolling(window=15, min_periods=1).apply(lambda x: x[x!= 0].mean()) # short term feature
    df['hr_30min_rolling'] = df['heart_rate'].rolling(window=30, min_periods=1).apply(lambda x: x[x!= 0].mean()) # long term feature
    df['hr_60min_rolling'] = df['heart_rate'].rolling(window=60, min_periods=1).apply(lambda x: x[x!= 0].mean()) # long term feature
    
    ## from scaled values
    #df_hr['hrscaled_5min_rolling'] = df_hr['hr_scaled'].rolling(window=5, min_periods=1).apply(lambda x: x[x!= 0].mean()) # short term feature
    #df_hr['hrscaled_10min_rolling'] = df_hr['hr_scaled'].rolling(window=10, min_periods=1).apply(lambda x: x[x!= 0].mean()) # short term feature
    #df_hr['hrscaled_30min_rolling'] = df_hr['hr_scaled'].rolling(window=30, min_periods=1).apply(lambda x: x[x!= 0].mean()) # long term feature
    
    return df

#############################################################################################################################################

def hrv_features(df):
    
    '''
    Returns a dataframe with heart rate variability engineered features.
    '''
    
    # from centred values
    df['last_measured_hrv_15min_centred'] = df['hrv_centred'].fillna(method="ffill", limit=15)
    df['hrv_abs_deviation_from_median_centred'] = abs(df['last_measured_hrv_15min_centred']) # absolute deviation from median,to account for atypical hrv changes to hypoglycemia
    df['hrv_change_15min_centred'] = df['last_measured_hrv_15min_centred'].diff(periods=15)
    df['abs_hrv_change_15min_centred'] = abs(df['hrv_change_15min_centred'])
    
    # from original values
    df['last_measured_hrv_15min'] = df['heart_rate_variability'].fillna(method="ffill", limit=15)
    df['hrv_change_15min'] = df['last_measured_hrv_15min'].diff(periods=15)
    df['abs_hrv_change_15min'] = abs(df['hrv_change_15min'])
    
    return df
    
#############################################################################################################################################

def bl_ox_rr_features(df):
    
    '''
    Returns dataframe with engineered features for blood oxygen saturation and respiratory rate.
    '''
    
    ## Blood oxygen saturation
    X_sorted1 = df['blood_oxygen_saturation'].dropna().sort_values(ascending=True) #dropping NaNs and sorting values to calculate the median
    median_value1 = statistics.median(X_sorted1)
    df['bl_ox_centred'] = df['blood_oxygen_saturation'] - median_value1
    df['last_measured_ox_centred'] = df['bl_ox_centred'].fillna(method="ffill", limit=30)
    
    # from original values
    df['last_measured_ox'] = df['blood_oxygen_saturation'].fillna(method="ffill", limit=30)

    #df['ox_pct_change'] = df['last_measured_ox'].pct_change(periods=30)
    
    ## respiratory rate
    X_sorted2 = df['respiratory_rate'].dropna().sort_values(ascending=True) #dropping NaNs and sorting values to calculate the median
    median_value2 = statistics.median(X_sorted2)
    df['rr_centred'] = df['respiratory_rate'] - median_value2
    df['last_measured_rr_centred'] = df['rr_centred'].fillna(method="ffill", limit=15)

    # from original values
    df['last_measured_rr'] = df['respiratory_rate'].fillna(method="ffill", limit=15)
       
    #df['rr_pct_change'] = df['last_measured_rr'].pct_change(periods=15)

    return df

#############################################################################################################################################

def steps_activity_features(df):
    
    '''
    Returns dataframe with step count and active energy engineered features.
    '''
    # fill NaNs with zeroes
    df['step_count'] = df['step_count'].fillna(0)
    df['active_energy'] = df['active_energy'].fillna(0)
    
    df['step_count_rollingsum_5min'] = df['step_count'].rolling(window=5, min_periods=1).apply(lambda x: x[x!= 0].sum()) # short term
    df['step_count_rollingsum_10min'] = df['step_count'].rolling(window=10, min_periods=1).apply(lambda x: x[x!= 0].sum()) # short term
    df['step_count_rollingsum_30min'] = df['step_count'].rolling(window=30, min_periods=1).apply(lambda x: x[x!= 0].sum()) # medium term
    df['step_count_rollingsum_60min'] = df['step_count'].rolling(window=60, min_periods=1).apply(lambda x: x[x!= 0].sum()) # long term

    df['active_energy_rollingsum_5min'] = df['active_energy'].rolling(window=5, min_periods=1).apply(lambda x: x[x!= 0].sum()) # short term
    df['active_energy_rollingsum_10min'] = df['active_energy'].rolling(window=10, min_periods=1).apply(lambda x: x[x!= 0].sum()) # short term
    df['active_energy_rollingsum_30min'] = df['active_energy'].rolling(window=30, min_periods=1).apply(lambda x: x[x!= 0].sum()) # medium term
    df['active_energy_rollingsum_60min'] = df['active_energy'].rolling(window=60, min_periods=1).apply(lambda x: x[x!= 0].sum()) # long term

    #df['step_count_15min_ago'] = df['step_count_rollingsum_5min'].shift(15) # lagging feature
    #df['step_count_30min_ago'] = df['step_count_rollingsum_5min'].shift(30) # lagging feature
    #df['step_count_60min_ago'] = df['step_count_rollingsum_5min'].shift(60) # lagging feature

    return df
    
#############################################################################################################################################

def meal_features(df):
    
    '''
    Returns dataframe with meal engineered features.
    '''
    
    # creating temp column to do further steps
    df['meal_temp'] = df['meal'].fillna(method="ffill")
    
    # reseting with a 0 every time a meal is ingested
    df['meal_temp'] = np.where(df['meal']==1, 0, df['meal_temp'])

    # cumulative sum that resets at the beginning of each meal ie at 0
    reset_mask = df['meal_temp'] == 0
    groups = reset_mask.cumsum()
    
    #calculating time since last meal 
    df['time_since_lastmeal'] = df.groupby(groups)['meal_temp'].cumsum()
    
    # creating an ordinal feature that refelcts how long it has been since the last meal 

    labels = [1,2,3,4,5,6]
    conditions =[
        (df['time_since_lastmeal'] >= 0) & (df['time_since_lastmeal'] <= 15),
        (df['time_since_lastmeal'] > 15) & (df['time_since_lastmeal'] <= 30),
        (df['time_since_lastmeal'] > 30) & (df['time_since_lastmeal'] <= 60),
        (df['time_since_lastmeal'] > 60) & (df['time_since_lastmeal'] <= 120),
        (df['time_since_lastmeal'] > 120) & (df['time_since_lastmeal'] <= 180),
        df['time_since_lastmeal'] > 180
        ]

    df['time_since_lastmeal_ord'] = np.select(conditions, labels, default= np.NaN)
    
    # dropping temp column
    df.drop('meal_temp', axis=1, inplace=True)

    
    return df

#############################################################################################################################################

def insulin_features(df):
    
    '''
    Returns dataframe with Insulin engineered features
    '''
    # subset short acting insulin column
    df_insulin = pd.DataFrame(df[['insulin_short']])

    # create a column for each short acting insulin shot, in case there are shots less than 4 hours apart
    counter = 0

    for index, row in df_insulin[['insulin_short']].iterrows():
        value = row['insulin_short']
        if value > 0:
            column_name = 'insulin_short_' + str(counter)
            df_insulin[column_name] = np.where(df_insulin.index == index, value, np.NaN)
            counter+=1

    # calculating Insulin on board
    df_ins = df_insulin.drop('insulin_short', axis=1)

    for column in df_ins.columns:
        value = df_ins[column].loc[df_ins[column].first_valid_index()]
        idx = df_ins[column].first_valid_index()
        df_ins[column].loc[idx :(idx+timedelta(minutes=60))] = 1
        df_ins[column].loc[(idx+timedelta(minutes=60)):(idx+timedelta(minutes=120))] = 0.75
        df_ins[column].loc[(idx+timedelta(minutes=120)):(idx+timedelta(minutes=180))] = 0.5
        df_ins[column].loc[(idx+timedelta(minutes=180)):(idx+timedelta(minutes=240))] = 0.25

    # summing across each row
    df_ins['IOB'] = df_ins.sum(axis=1)
    
    df['IOB'] = df_ins['IOB']
    
    return df
    
#############################################################################################################################################

def glucose_features(df):

    '''
    Returns dataframe with cgm glucose features
    '''
    
    # fill in between CGM readings
    df['cgm'] = df['glucose'].fillna(method='bfill', limit=5)
    df['hypoglycemia_temp'] = np.where(df['cgm'] < 4, 1, 0).astype('int')
    df['hypoglycemia_temp'] = np.where(df['cgm'].isna(), np.NaN, df['hypoglycemia_temp']) # making sure NaNs remain NaNs
    
    # calculating duration of hypoglycemic episodes, to discard anything below 15 minutes
    df["hypo_duration"] = df["hypoglycemia_temp"][df.hypoglycemia_temp.notna()].groupby((df["hypoglycemia_temp"] == 0).cumsum()).cumcount()
    df['hypo_duration_15more'] = np.where((df["hypo_duration"] < 15) | (df["hypo_duration"].isna()), np.NaN, 1)
    df['hypo_duration_15more'] = df['hypo_duration_15more'].fillna(method='bfill', limit=14)
    df['hypo_duration_15more'] = df['hypo_duration_15more'].fillna(0).astype('int')
    df['hypo_duration_15more'] = np.where(df['cgm'].isna(), np.NaN, df['hypo_duration_15more']) # making sure NaNs remain NaNs
    df['hypoglycemia'] = df['hypo_duration_15more']
    
    # creating a feature that captures the 1st 60 minutes of a hypoglycemic event
    #df['hypoglycemia_90min'] =  df['hypoglycemia'][df.hypoglycemia.notna()].groupby((df["hypoglycemia"] == 0).cumsum()).cumcount()
    #df['hypoglycemia_90min'] = np.where((df['hypoglycemia_90min'] <= 90) & (df['hypoglycemia_90min'] >0), 1, 0)
    #df['hypoglycemia_90min'] = np.where(df['cgm'].isna(), np.NaN, df['hypoglycemia_90min']) # making sure NaNs remain NaNs
    
    # marking severe hypoglycemia events to possibly consider them separately (the body's response is different during severe hypoglycemia)
    #df['severe_hypoglycemia'] =  np.where(df['cgm'] < 2.8, 1, 0)
    #df['severe_hypoglycemia'] = np.where(df['cgm'].isna(), np.NaN, df['severe_hypoglycemia']) # making sure NaNs remain NaNs

    # creating a column for prehypoglycemia phase (30 minutes before the start of hypoglycemia)
    df['hypo_shift'] = df['hypo_duration_15more'].shift(-1)
    df['hypo_start'] = np.where((df['hypo_duration_15more'] == 0) & (df['hypo_shift'] == 1), 1, np.NaN)
    df['hypo_end'] = np.where((df['hypo_duration_15more'] == 1) & (df['hypo_shift'] == 0), 1, np.NaN)
    df['prehypoglycemia'] = df['hypo_start'].fillna(method='bfill', limit=60)
    df['prehypoglycemia'] = df['prehypoglycemia'].fillna(0).astype('int')
    df['prehypoglycemia'] = np.where(df['cgm'].isna(), np.NaN, df['prehypoglycemia']) # making sure NaNs remain NaNs

    
    return df
    
#############################################################################################################################################

def feature_engineering(filename):
    '''
    Collective function for all feature engineering functions in addition to adding 'hour' and 'patient_code' features
    '''

    # read the data
    df = pd.read_csv(filename)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    ## Heart rate
    hr_centred = create_centred_metric(df, 'heart_rate')
    df['hr_centred'] = hr_centred
    df = hr_rolling_features(df)

    ## Heart rate variability
    hrv_centred = create_centred_metric(df, 'heart_rate_variability')
    df['hrv_centred'] = hrv_centred
    df = hrv_features(df)

    ## bl ox and rr
    df = bl_ox_rr_features(df)

    ## steps and activity
    df = steps_activity_features(df)

    ## meal
    df = meal_features(df)

    ## insulin
    df = insulin_features(df)

    ## cgm
    df = glucose_features(df)

    ## time
    df['hour'] = df.index.hour

    ## patient code
    df['patient_code'] = filename[:5]
    
    return df

#############################################################################################################################################