import pandas as pd
import numpy as np
import pickle
from Data_collecter import weather_history

def feature_gen(date, save =True):
    df_weather = weather_history(date_go_back= date)


    # convert to int
    features = ['WindChillC', 'humidity', 'pressure', 'temperature', 'wind_speed', 'time']

    for feat in features:
        df_weather[feat] = df_weather[feat].apply(lambda x: int(x))

    features = ['WindChillC', 'humidity', 'pressure', 'temperature', 'wind_speed']

    for f in features:
        for x in range(1, 25):
            col_name = f + '_B' + str(x)
            df_weather[col_name] = df_weather[f].shift(x)

    # sums and means
    for f in features:
        for s in range(1, 25):
            col_name = f + '_mean' + str(s)
            df_weather[col_name] = df_weather[f].rolling(s).mean()

            col_name = f + '_sum' + str(s)
            df_weather[col_name] = df_weather[f].rolling(s).sum()

    # differences for all means and sums
    A = [a for a in range(1, 25)]
    B = [b for b in range(1, 25)]
    cartisian_list = [(a, b) for a in A for b in B if (a != b) and (a < b)]
    for f in features:
        for diff in cartisian_list:
            col_name = f + '_mean_diff_' + str(diff[0]) + '_' + str(diff[1])
            df_weather[col_name] = df_weather[f + '_mean' + str(diff[0])] - df_weather[f + '_mean' + str(diff[1])]

            col_name = f + '_sum_diff_' + str(diff[0]) + '_' + str(diff[1])
            df_weather[col_name] = df_weather[f + '_sum' + str(diff[0])] - df_weather[f + '_sum' + str(diff[1])]
    if save:
        file_name = 'Data//df_weather.p'
        pickled = open(file_name, 'wb')
        pickle.dump(df_weather,pickled)
        pickled.close()
