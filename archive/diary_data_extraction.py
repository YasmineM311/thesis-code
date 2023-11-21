import pandas as pd


# defining a function for manually inserted data preprocessing

def preprocess_diary(file):
    '''
    loads manually documented data by the patient (food, exercise and 
    Insulin diary) and preprocesses it'''
    
    # load the data
    diary = pd.read_excel(file)
    
    # fix the timestamp
    diary['timestamp'] = diary['timestamp'].str.replace('T' , ' ')
    
    # coerce it to be of datatype datetime
    diary['timestamp'] = pd.to_datetime(diary['timestamp'])
    
    # set the timestamp to be the index
    diary.set_index('timestamp', inplace=True)
    
    # drop unnecessary columns
    # diary.drop('meal', axis=1, inplace=True)
    return diary