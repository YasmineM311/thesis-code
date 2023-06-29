import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from pyod.models.ecod import ECOD


def impute_applewatch_data(df):
    '''
    This function preprocesses and imputes apple watch data.

    1- Heart rate: 
    Outliers are identified and removed, in our use case outliers are any recordings that occur only once.
    Then the number of consecutive NaNs is calculated to mask entries preceeded by more than 20 NaNs from being imputed,
    as these represent periods where the participant took off the watch.
    Lastly, imputation is performed using interpolation.

    2- Step count:
    NaNs are replaced with zeroes with the exception of time points where the watch was not worn by the participant.

    3- Heart rate variability:
    Imputation is doine using the KNN imputer using selected columns. As with other metrics, time periods where the 
    participant took off the watch are not imputed and remain NaNs.

    4- Respiratory rate and blood oxygen saturation
    These metrics are recorded only during sleep and are only imputed (by interpolation) during sleep.
    '''
    ####################################################################################################################
    ## Heart Rate 
    # first we remove outliers:
    # 1. Round heart rate recordings
    #df['heart_rate'] = df['heart_rate'].round()

    # 2. Detect outliers and replace them with NaNs
    Y = np.array(df.heart_rate[df.heart_rate > 0])
    Y = Y.reshape(-1, 1)
    clf = ECOD(contamination=0.001)
    clf.fit(Y)
    outliers = clf.predict(Y)
    outlier_list = Y[np.where(outliers==1)]
    if len(outlier_list) > 0:
        outlier_list = np.concatenate(outlier_list).ravel().tolist()
    else:
        outlier_list = []
    
    df['heart_rate'] = np.where(df['heart_rate'].isin(outlier_list), np.NaN, df['heart_rate'])

    # then we interpolate making sure to mask any entries preceeded by more than 20 NaNs from being imputed:
    # 1. Determine the number of consecutive NaNs in each column
    consecutive_nans = df.apply(lambda x: x.isnull().astype(int).groupby(x.notnull().astype(int).cumsum()).cumsum())
    
    # 2. Create a mask to exclude anything with more than 20 consecutive NaNs
    hr_mask = consecutive_nans['heart_rate'] < 4 

    # 3. Initiate imputed column 
    df['heart_rate_interpolated'] = np.NaN
    
    # 4. Interpolated only masked rows
    df['heart_rate_interpolated'] = df['heart_rate_interpolated'].mask(hr_mask, df['heart_rate'].interpolate())

    #####################################################################################################################
    ## Step count
    # 1. fill NaNs with 0 for the step count column
    #df['steps_imputed'] = df['step_count'].fillna(0).round()

    # 2. Create a boolean mask where other columns have NaN values (watch was not worn by the participant)
    mask = df[['heart_rate_interpolated', 'heart_rate_variability', 'step_count', 'active_energy']].isna().all(axis=1)

    # 3. then we coerce empty rows to NaNs
    #df['steps_imputed'] = df['steps_imputed'].mask(mask, np.nan) 
    
    #####################################################################################################################
    ## Heart Rate Variability
    # 1. Detect outliers and replace them with NaNs
    Y = np.array(df.heart_rate_variability[df.heart_rate_variability > 0])
    Y = Y.reshape(-1, 1)
    clf = ECOD(contamination=0.005)
    clf.fit(Y)
    outliers = clf.predict(Y)
    outlier_list = Y[np.where(outliers==1)]
    if len(outlier_list) > 0:
        outlier_list = np.concatenate(outlier_list).ravel().tolist()
    else:
        outlier_list = []
    
    df['heart_rate_variability'] = np.where(df['heart_rate_variability'].isin(outlier_list), np.NaN, df['heart_rate_variability'])
    
    # 2. creating a rolling average for hrv
    #df['hrv_rolling_15'] = df['heart_rate_variability'].rolling(window=15, min_periods=1).apply(lambda x: x[x!= 0].mean())
    #df['hrv_rolling_5'] = df['heart_rate_variability'].rolling(window=5, min_periods=1).apply(lambda x: x[x!= 0].mean())
    #df['hrv_fill'] = df['heart_rate_variability'].fillna(method='ffill', limit=1).fillna(method='bfill', limit=1)
    
    #marking imputed values
    df['hrv_imputed'] = np.where(df['heart_rate_variability'] > 0, 0, 1)

    # 3. subsetting the metrics that we want to use for imputation
    df_sleep = pd.DataFrame(df[df.sleep == 1])
    df_awake = pd.DataFrame(df[df.sleep == 0])
    
    #hrv_impute_sleep_df = df_sleep[['heart_rate_interpolated', 'hrv_fill']]
    #hrv_impute_awake_df = df_awake[['heart_rate_interpolated', 'hrv_fill']]
    hrv_impute_sleep_df = df_sleep[['heart_rate_interpolated', 'heart_rate_variability']]
    hrv_impute_awake_df = df_awake[['heart_rate_interpolated', 'heart_rate_variability']]
    

    # 4. initiating the KNN imputer 
    imputer = KNNImputer(n_neighbors=10)
    imputed_values_sleep = imputer.fit_transform(hrv_impute_sleep_df)
    imputed_values_awake = imputer.fit_transform(hrv_impute_awake_df)

    # 5. subsetting hrv imputed values
    hrv_imputed_values_sleep = imputed_values_sleep[:, -1]
    hrv_imputed_values_awake = imputed_values_awake[:, -1]

    # 6. assign the values to a new column
    #df['hrv_rolling_15_imputed'] = hrv_imputed_values.round()
    #df['hrv_rolling_5_imputed'] = hrv_imputed_values.round(2)
    df_sleep['hrv_knn_imp'] = hrv_imputed_values_sleep.round(2)
    
    df_awake['hrv_knn_imp'] = hrv_imputed_values_awake.round(2)
    
    df = pd.concat([df_sleep, df_awake], axis=0)
    df.sort_values(by='timestamp', inplace=True)

    # 7. then we coerce empty rows to NaNs (watch was not worn by the participant)
    df['hrv_knn_imp'] = df['hrv_knn_imp'].mask(mask, np.nan)

    #####################################################################################################################
    ## Respiratory rate and blood oxygen saturation
    # 1.Create a boolean mask for sleep values only
    night_mask = (df['sleep'] == 1)
    
    # 2.initiate a column for inerpolated rr
    df['rr_intrapolated'] = np.NaN
    df['bl_oxygen_interpolated'] = np.NaN

    # 3.applying the mask to ensure interpolation occurs only during sleep
    #df['rr_intrapolated'] = df['rr_intrapolated'].mask(night_mask, df['respiratory_rate'].interpolate()).round(2)
    #df['bl_oxygen_interpolated'] = df['bl_oxygen_interpolated'].mask(night_mask, df['blood_oxygen_saturation'].interpolate()).round(2)

    df['rr_intrapolated'] = df['rr_intrapolated'].mask(night_mask, df['respiratory_rate'].ffill())
    df['bl_oxygen_interpolated'] = df['bl_oxygen_interpolated'].mask(night_mask, df['blood_oxygen_saturation'].ffill())
    #####################################################################################################################

    return df

###################################################################################################################################################

# backward filling of cgm data

def cgm_data_backward_fill(df):
    df['glucose_imputed'] = df['glucose'].fillna(method='bfill')
    return df