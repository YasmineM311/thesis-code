# backward filling of cgm data

def cgm_data_backward_fill(df):
    df['glucose_imputed'] = df['glucose'].fillna(method='bfill',limit=10) # we set the limit to 10 to avoid imputing when there are gaps in the data
    return df