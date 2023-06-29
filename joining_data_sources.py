import pandas as pd

#def join_data_sources(df1, df2, df3):
'''
    This function merges all three data sources where:
    df1: is apple watch data
    df2: is cgm data
    df3: is diary data
    '''

    # we use merge_ordered to join data in the order of the timestamp 

  #  df_merged1 = pd.merge_ordered(df1, df2, on='timestamp')
  #  df_merged2 = pd.merge_ordered(df_merged1, df3, on='timestamp')

   # return df_merged2

def join_data_sources(df1, df2):
    '''
    This function merges all three data sources where:
    df1: is apple watch data
    df2: is cgm data
    '''

    # we use merge_ordered to join data in the order of the timestamp 

    #df_merged1 = pd.merge_ordered(df1, df2, on='timestamp')
    df_merged1 = pd.merge_ordered(df1, df2, on='timestamp')

    return df_merged1