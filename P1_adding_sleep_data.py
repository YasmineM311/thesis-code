# specifying sleep timeframes to create a binary 'sleep' column

df['sleep'] = np.where(((df['timestamp'] > '2023-05-11 23:45:00') & (df['timestamp'] < '2023-05-12 06:20:00'))
                        |
                       ((df['timestamp'] > '2023-05-12 23:45:00') & (df['timestamp'] < '2023-05-13 08:45:00'))
                        |
                       ((df['timestamp'] > '2023-05-14 01:00:00') & (df['timestamp'] < '2023-05-14 08:30:00'))
                        |
                       ((df['timestamp'] > '2023-05-14 23:45:00') & (df['timestamp'] < '2023-05-15 06:15:00'))
                        |
                       ((df['timestamp'] > '2023-05-15 23:30:00') & (df['timestamp'] < '2023-05-16 06:15:00'))
                        |
                       ((df['timestamp'] > '2023-05-17 00:00:00') & (df['timestamp'] < '2023-05-17 06:15:00'))
                        |
                       ((df['timestamp'] > '2023-05-18 00:45:00') & (df['timestamp'] < '2023-05-18 08:30:00'))
                        |
                       ((df['timestamp'] > '2023-05-19 00:45:00') & (df['timestamp'] < '2023-05-19 09:20:00'))
                       |
                       ((df['timestamp'] > '2023-05-19 23:00:00') & (df['timestamp'] < '2023-05-20 06:15:00')), 1, 0)
                               