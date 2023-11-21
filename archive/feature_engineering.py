import pandas as pd
import numpy as np


def feature_engineering(file): 
    
    # read the aw/cgm/sleep data and set timestamp to be index
        
    df = pd.read_csv(file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    ############################################### SELECT METRICS IMPUTATION ####################################################

    df['step_count'] = df['step_count'].fillna(0)
    df['active_energy'] = df['active_energy'].fillna(0)
    night_mask = (df['sleep'] == 1)
    df['rr_ffill'] = df['respiratory_rate'].mask(night_mask, df['respiratory_rate'].ffill())
    df['bl_oxygen_ffill'] = df['blood_oxygen_saturation'].mask(night_mask, df['blood_oxygen_saturation'].ffill())


    ################################################## ROLLING STATISTICS ########################################################

    # heart rate
    df['hr_rolling_15'] = df['heart_rate'].rolling(window=15, min_periods=1).apply(lambda x: x[x!= 0].mean())
    df['hr_rolling_30'] = df['heart_rate'].rolling(window=30, min_periods=1).apply(lambda x: x[x!= 0].mean())
    df['hr_rolling_45'] = df['heart_rate'].rolling(window=45, min_periods=1).apply(lambda x: x[x!= 0].mean())
    df['hr_rolling_60'] = df['heart_rate'].rolling(window=60, min_periods=1).apply(lambda x: x[x!= 0].mean())

    # heart rate variability
    df['hrv_rolling_15'] = df['heart_rate_variability'].rolling(window=15, min_periods=1).apply(lambda x: x[x!= 0].mean())
    df['hrv_rolling_30'] = df['heart_rate_variability'].rolling(window=30, min_periods=1).apply(lambda x: x[x!= 0].mean())
    df['hrv_rolling_45'] = df['heart_rate_variability'].rolling(window=45, min_periods=1).apply(lambda x: x[x!= 0].mean())
    df['hrv_rolling_60'] = df['heart_rate_variability'].rolling(window=60, min_periods=1).apply(lambda x: x[x!= 0].mean())

    # steps
    df['steps_rolling_15'] = df['step_count'].rolling(window=15, min_periods=1).apply(lambda x: x[x!= 0].sum())
    df['steps_rolling_30'] = df['step_count'].rolling(window=30, min_periods=1).apply(lambda x: x[x!= 0].sum())
    df['steps_rolling_45'] = df['step_count'].rolling(window=45, min_periods=1).apply(lambda x: x[x!= 0].sum())
    df['steps_rolling_60'] = df['step_count'].rolling(window=60, min_periods=1).apply(lambda x: x[x!= 0].sum())

    # active energy
    df['active_energy_15'] = df['active_energy'].rolling(window=15, min_periods=1).apply(lambda x: x[x!= 0].sum())
    df['active_energy_30'] = df['active_energy'].rolling(window=30, min_periods=1).apply(lambda x: x[x!= 0].sum())
    df['active_energy_45'] = df['active_energy'].rolling(window=45, min_periods=1).apply(lambda x: x[x!= 0].sum())
    df['active_energy_60'] = df['active_energy'].rolling(window=60, min_periods=1).apply(lambda x: x[x!= 0].sum())

    # respiratory rate
    df['rr_rolling_10'] = df['rr_ffill'].rolling(window=10, min_periods=1).apply(lambda x: x[x!= 0].mean())
    df['rr_rolling_20'] = df['rr_ffill'].rolling(window=20, min_periods=1).apply(lambda x: x[x!= 0].mean())
    df['rr_rolling_30'] = df['rr_ffill'].rolling(window=30, min_periods=1).apply(lambda x: x[x!= 0].mean())

    
    ############################################ HR/HRV DEVIATION FROM MEDIAN ######################################################

    mask_sleep = (df.sleep == 1)
    mask_awake_active = (df.sleep==0) & (df.active_energy > 0)
    mask_awake_resting = (df.sleep==0) & (df.active_energy == 0)

    # HR agg statistics
    #hr_sleep_mean = df[df.sleep==1].heart_rate.mean()
    hr_sleep_median = df[df.sleep==1].heart_rate.median()
    hr_sleep_std = df[df.sleep==1].heart_rate.std()

    #hr_awake_active_mean = df[(df.sleep==0) & (df.active_energy > 0)].heart_rate.mean()
    hr_awake_active_median = df[(df.sleep==0) & (df.active_energy > 0)].heart_rate.mean()
    hr_awake_active_std = df[(df.sleep==0) & (df.active_energy > 0)].heart_rate.std()

    #hr_awake_resting_mean = df[(df.sleep==0) & (df.active_energy == 0)].heart_rate.mean()
    hr_awake_resting_median = df[(df.sleep==0) & (df.active_energy == 0)].heart_rate.mean()
    hr_awake_resting_std = df[(df.sleep==0) & (df.active_energy == 0)].heart_rate.std()

    # HR deviation from median 
    df['hr15_dev_from_median'] = np.NaN
    df['hr30_dev_from_median'] = np.NaN
    df['hr45_dev_from_median'] = np.NaN
    df['hr60_dev_from_median'] = np.NaN

    df['hr15_dev_from_median'] = df['hr15_dev_from_median'].mask(mask_sleep, df['hr_rolling_15'] - hr_sleep_median)
    df['hr15_dev_from_median'] = df['hr15_dev_from_median'].mask(mask_awake_active, df['hr_rolling_15'] - hr_awake_active_median)
    df['hr15_dev_from_median'] = df['hr15_dev_from_median'].mask(mask_awake_resting, df['hr_rolling_15'] - hr_awake_resting_median)

    df['hr30_dev_from_median'] = df['hr30_dev_from_median'].mask(mask_sleep, df['hr_rolling_30'] - hr_sleep_median)
    df['hr30_dev_from_median'] = df['hr30_dev_from_median'].mask(mask_awake_active, df['hr_rolling_30'] - hr_awake_active_median)
    df['hr30_dev_from_median'] = df['hr30_dev_from_median'].mask(mask_awake_resting, df['hr_rolling_30'] - hr_awake_resting_median)

    df['hr45_dev_from_median'] = df['hr45_dev_from_median'].mask(mask_sleep, df['hr_rolling_45'] - hr_sleep_median)
    df['hr45_dev_from_median'] = df['hr45_dev_from_median'].mask(mask_awake_active, df['hr_rolling_45'] - hr_awake_active_median)
    df['hr45_dev_from_median'] = df['hr45_dev_from_median'].mask(mask_awake_resting, df['hr_rolling_45'] - hr_awake_resting_median)

    df['hr60_dev_from_median'] = df['hr60_dev_from_median'].mask(mask_sleep, df['hr_rolling_60'] - hr_sleep_median)
    df['hr60_dev_from_median'] = df['hr60_dev_from_median'].mask(mask_awake_active, df['hr_rolling_60'] - hr_awake_active_median)
    df['hr60_dev_from_median'] = df['hr60_dev_from_median'].mask(mask_awake_resting, df['hr_rolling_60'] - hr_awake_resting_median)

    # HRV agg statistics
    #hrv_sleep_mean = df[df.sleep==1].hrv_rolling_15.mean()
    hrv_sleep_median = df[df.sleep==1].heart_rate_variability.median()
    hrv_sleep_std = df[df.sleep==1].heart_rate_variability.std()

    #hrv_awake_active_mean = df[(df.sleep==0) & (df.active_energy > 0)].hrv_rolling_15.mean()
    hrv_awake_active_median = df[(df.sleep==0) & (df.active_energy > 0)].heart_rate_variability.mean()
    hrv_awake_active_std = df[(df.sleep==0) & (df.active_energy > 0)].heart_rate_variability.std()

    #hrv_awake_resting_mean = df[(df.sleep==0) & (df.active_energy == 0)].hrv_rolling_15.mean()
    hrv_awake_resting_median = df[(df.sleep==0) & (df.active_energy == 0)].heart_rate_variability.mean()
    hrv_awake_resting_std = df[(df.sleep==0) & (df.active_energy == 0)].heart_rate_variability.std()

    # HRV deviation from median
    df['hrv15_dev_from_median'] = np.NaN
    df['hrv30_dev_from_median'] = np.NaN
    df['hrv45_dev_from_median'] = np.NaN
    df['hrv60_dev_from_median'] = np.NaN
    
    df['hrv15_dev_from_median'] = df['hrv_dev_from_median'].mask(mask_sleep, df['hrv_rolling_15'] - hrv_sleep_median)
    df['hrv15_dev_from_median'] = df['hrv_dev_from_median'].mask(mask_awake_active, df['hrv_rolling_15'] - hrv_awake_active_median)
    df['hrv15_dev_from_median'] = df['hrv_dev_from_median'].mask(mask_awake_resting, df['hrv_rolling_15'] - hrv_awake_resting_median)

    df['hrv30_dev_from_median'] = df['hrv_dev_from_median'].mask(mask_sleep, df['hrv_rolling_30'] - hrv_sleep_median)
    df['hrv30_dev_from_median'] = df['hrv_dev_from_median'].mask(mask_awake_active, df['hrv_rolling_30'] - hrv_awake_active_median)
    df['hrv30_dev_from_median'] = df['hrv_dev_from_median'].mask(mask_awake_resting, df['hrv_rolling_30'] - hrv_awake_resting_median)

    df['hrv45_dev_from_median'] = df['hrv_dev_from_median'].mask(mask_sleep, df['hrv_rolling_45'] - hrv_sleep_median)
    df['hrv45_dev_from_median'] = df['hrv_dev_from_median'].mask(mask_awake_active, df['hrv_rolling_45'] - hrv_awake_active_median)
    df['hrv45_dev_from_median'] = df['hrv_dev_from_median'].mask(mask_awake_resting, df['hrv_rolling_45'] - hrv_awake_resting_median)

    df['hrv60_dev_from_median'] = df['hrv_dev_from_median'].mask(mask_sleep, df['hrv_rolling_60'] - hrv_sleep_median)
    df['hrv60_dev_from_median'] = df['hrv_dev_from_median'].mask(mask_awake_active, df['hrv_rolling_60'] - hrv_awake_active_median)
    df['hrv60_dev_from_median'] = df['hrv_dev_from_median'].mask(mask_awake_resting, df['hrv_rolling_60'] - hrv_awake_resting_median)


    ##################################################### PERCENTAGE CHANGE ###########################################################


    # heart rate
    df['hr_rolling_15_pct'] = df['hr_rolling_15'].pct_change(periods=15, fill_method=None).round(2) 
    df['hr_rolling_30_pct'] = df['hr_rolling_30'].pct_change(periods=30, fill_method=None).round(2) 
    df['hr_rolling_45_pct'] = df['hr_rolling_45'].pct_change(periods=45, fill_method=None).round(2) 
    df['hr_rolling_60_pct'] = df['hr_rolling_60'].pct_change(periods=60, fill_method=None).round(2) 

    # heart rate variability
    df['hrv_rolling_15_pct'] = df['hrv_rolling_15'].pct_change(periods=15, fill_method=None).round(2) 
    df['hrv_rolling_30_pct'] = df['hrv_rolling_30'].pct_change(periods=30, fill_method=None).round(2) 
    df['hrv_rolling_45_pct'] = df['hrv_rolling_45'].pct_change(periods=45, fill_method=None).round(2) 
    df['hrv_rolling_60_pct'] = df['hrv_rolling_60'].pct_change(periods=60, fill_method=None).round(2) 

    # respiratory rate
    df['rr_pct_10'] = df['rr_rolling_10'].pct_change(periods=10, fill_method=None).round(2) 
    df['rr_pct_20'] = df['rr_rolling_20'].pct_change(periods=20, fill_method=None).round(2) 
    df['rr_pct_30'] = df['rr_rolling_30'].pct_change(periods=30, fill_method=None).round(2) 

   
    ########################################## SLEEP HRV DIFFERENCING FROM LAST NON-NULL VALUE ###########################################

    # heart rate variability sleep mask
    consecutive_nans = df.apply(lambda x: x.isnull().astype(int).groupby(x.notnull().astype(int).cumsum()).cumsum())
    mask = consecutive_nans['heart_rate_variability'] < 20 

    # differencing closely measured non NaN values
    df_masked = df[mask]
    df_masked['heart_rate_variability_diff'] = df_masked['heart_rate_variability'].diff()
    df['hrv_diff'] = df_masked['heart_rate_variability_diff']

    # sum of hrv difference in last 30 min
    df['hrv30_diff_sum'] = df['hrv_diff'].rolling(window=30, min_periods=1).sum()
    df['hrv30_diff_sum'] = np.where(df.heart_rate_variability > 0, df['hrv30_diff_sum'], np.NaN)

    # absolute sum of hrv difference in last 30 min
    df['hrv30_diff_abs_sum'] = df['hrv_diff'].rolling(window=30, min_periods=1).apply(lambda x: abs(x[x!= 0]).sum())
    df['hrv30_diff_abs_sum'] = np.where(df.heart_rate_variability > 0, df['hrv30_diff_abs_sum'], np.NaN)


    ########################################## CGM glucose backfill and hypo/prehypo binary columns ###########################################

    df['glucose'] = df['glucose'].fillna(method='bfill', limit=5)
    df['hypoglycemia'] = np.where(df['glucose'] <= 3.9, 1, 0)
    df['prehypoglycemia'] = df['hypoglycemia'].shift(-30)
    df['prehypoglycemia'] = np.where((df['prehypoglycemia'] == 1) & (df['hypoglycemia'] == 0), 1, 0)


    ###########################################################################################################################################

    return df