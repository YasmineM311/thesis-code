import pandas as pd
import os


path = 'C:\\Users\\y_mus\\OneDrive - Hochschule Luzern\\HSLU\\Thesis\\Patient_raw_data\\Participant_8'
os.chdir(path)

df = pd.read_csv('LUZ08_watch_cgm_sleep_data.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

df = df.resample('1T').mean()

df.to_csv('LUZ08_resampled.csv')

print('success')