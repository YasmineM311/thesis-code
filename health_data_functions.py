# import required modules
import json
import pandas as pd

# define required functions
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


def fetch_data_from_file(file_name):
    '''
    Extracts relevant metrics from JSON file
    containing health data, and combines them into 
    a data frame'''

    # reading data from file
    health_data = pd.read_json(file_name)

    # transforming the data using the transform function 
    # defined previously
    transformed = transform(health_data)

    # converting it into a data frame
    df = pd.DataFrame(transformed)

    # extracting relevant metrics using previously defined functions
    heart_rate_var = extract_metric(df, 'heart_rate_variability')
    active_energy = extract_metric(df, 'active_energy')
    respiratory_rate = extract_metric(df, 'respiratory_rate')
    step_count = extract_metric(df, 'step_count')
    heart_rate = extract_heart_rate(df)

    # define a list that contains the names of all metrics
    metrics = [heart_rate, heart_rate_var, active_energy, respiratory_rate, step_count]

    # combine all metrics into a data frame
    all_metrics = pd.DataFrame().join(metrics, how="outer")

    # sort data by date in ascending order
    all_metrics.sort_values(by='date', ascending = True, inplace = True)

    # replace NaNs with zeroes
    all_metrics_imputed = all_metrics.fillna(0)

    return all_metrics_imputed


