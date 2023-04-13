# import required modules
# import json
import pandas as pd
from datetime import datetime, timezone


# define required functions

# The following function flattens and extracts the json data to easily 
# convert it into a dataframe to facilitate downstream processes 
def transform(data):
    '''
    Flatten the nested JSON data structure giving each data point the
    name and unit information, while getting rid of metrics that have
    empty data points.
    '''
    # initiate empty list to add the data points to it 
    rows = []

    # flattening the json structure and fetching the 
    # name and unit for each metric
    for metric in data.get('data', {}).get('metrics', []):
        name = metric['name']
        units = metric['units']

    # adding name and unit information to each data 
    # point, so that each data point holds full information
        for point in metric.get('data', []):
            point['name'] = name
            point['units'] = units
            rows.append(point)

    return rows


# The following two functions extract the relevant health metrics
def extract_metric(df, metric_name):
    '''
    A function to extract data for a specific metric, whose name
    is given as a variable to the function.
    This function works does not work for heart_rate, for which
    there is a separate function.
    '''
    # extracting the metric
    metric = df[df.name==metric_name][['date', 'qty']]
    
    # changing the column name to reflect the metric
    metric.rename(columns={'qty':metric_name}, inplace=True)
    
    # changing the date column to be of datetime type
    metric['date'] = pd.to_datetime(metric['date'])
    
    # making the date column the index
    metric.set_index('date', inplace=True)
    
    return metric


def extract_heart_rate(df):
    '''
    A function to extract heart_rate data
    '''
    #extracting the metric
    heart_rate = df[df.name=='heart_rate'][['date', 'Avg']]
    
    #changing the column name to reflect the metric
    heart_rate.rename(columns={'Avg':'heart_rate'}, inplace=True)
    
    # changing the date column to be of datetime type
    heart_rate['date'] = pd.to_datetime(heart_rate['date'])
    
    # making the date column the index
    heart_rate.set_index('date', inplace=True)
    
    return heart_rate


# The following function extracts and creates new columns for
# day and time information from the timestamp to facilitate 
# joining with other data sources based on these two columns
def create_day_and_time_columns(df):
    '''
    A function to create two new columns "day" and "time"
    extracted from the timestamp to serve as a basis for 
    joining all data sources together
    '''
    # subfunction to remove timezone information from timestamp
    def remove_timezone(dt):   
        return dt.replace(tzinfo=None)

    # create a new column "timestamp" that mirrors the index
    df['timestamp'] = df.index

    # apply the remove_timezone subfunction to the timestamp column
    # direct attempts to remove timezone information from the column as a 
    # pandas series were not successful as the parameter UTC has to be
    # set to TRUE and that would mess up our timestamp 
    # hence we resort to the apply function
    df['timestamp'] = df['timestamp'].apply(remove_timezone)

    # now creating the new columns 'day' and 'time'
    df['day'] = df['timestamp'].dt.strftime('%d-%m-%Y')
    df['time'] = df['timestamp'].dt.strftime('%H:%M')

    return df


# This function stacks the previous functions and performs the following:
# 1-flattens json data from json file
# 2-converts it into a dataframe
# 3-extracts the relevant metrics
# 4-stacks the extracted metrics into one dataframe
# 5-sorts values in the combined dataframe by datetime
# 6-imputes NaNs into zeroes
# 7-creates day and time columns
# 8-returns final dataframe
def fetch_data_from_file(file_name):
    '''
    Extracts relevant metrics from Apple watch data JSON file,
    combines them into a data frame, 
    sorts values by timestamp,
    imputes NaNs and
    adds day and time columns
    Returns: A dataframe containing Apple Watch data ready to be combined with other data sources'''

    # reading data from file
    health_data = pd.read_json(file_name)

    # transforming the data using the transform function defined previously
    transformed = transform(health_data)

    # converting it into a data frame
    df = pd.DataFrame(transformed)

    # extracting relevant metrics using previously defined functions
    heart_rate_var = extract_metric(df, 'heart_rate_variability')
    active_energy = extract_metric(df, 'active_energy')
    respiratory_rate = extract_metric(df, 'respiratory_rate')
    step_count = extract_metric(df, 'step_count')
    blood_oxygen = extract_metric(df, 'blood_oxygen_saturation')
    heart_rate = extract_heart_rate(df)
    
    # define a list that contains the names of all metrics
    metrics = [heart_rate, heart_rate_var, active_energy, respiratory_rate, step_count, blood_oxygen]

    # combine all metrics into a data frame
    all_metrics_df = pd.DataFrame().join(metrics, how="outer")

    # sort data by date in ascending order
    all_metrics_df.sort_values(by='date', ascending = True, inplace = True)

    # replace NaNs with zeroes
    all_metrics_imputed = all_metrics_df.fillna(0)

    # create 'day' and 'time' columns
    final_df = create_day_and_time_columns(all_metrics_imputed)

    return final_df


