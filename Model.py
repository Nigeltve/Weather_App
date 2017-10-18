import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dateutil.relativedelta as RD

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from Feature_generator import feature_gen
from sklearn.metrics import r2_score

file_reg_path = Path('Data//regression.p')
weather_data_path = Path('Data//df_weather.p')

reg = RandomForestRegressor(n_estimators= 400,n_jobs=-1, max_depth= 45 , min_samples_split= 45, max_features= 700)

now = pd.datetime.today()                   # gets a date for the current day and converst it
standard_time_back =  now - RD.relativedelta(months=6)

now = now.strftime('%Y-%m-%d')
now = str(now)
standard_time_back = standard_time_back.strftime('%Y-%m-%d')
standard_time_back = str(standard_time_back)

needs_updating_flag = None

print('current time {}\ndefault time back {}\n'.format(now, standard_time_back))



def checking_weather_data():
    if weather_data_path.is_file():
        print('weather Data is present')
        print('checking if up to date')

        pickle_in = open(weather_data_path, 'rb')
        df_weather = pickle.load(pickle_in)
        df_weather.dropna(inplace=True)

        unique_period = df_weather.Date.unique().tolist()


        if now in unique_period:
            print('data is up to date')

            needs_updating_flag = False
        else:
            current_max_date = str(max(unique_period))
            print('Data is not up to date\nupdating')
            feature_gen(date=standard_time_back, save=True)

            pickle_in = open(weather_data_path, 'rb')
            df_weather = pickle.load(pickle_in)
            df_weather.dropna(inplace=True)

            needs_updating_flag = True

    return df_weather, needs_updating_flag

def checking_regression(passed_flag, passed_DataFrame):
    if file_reg_path.is_file():
        if needs_updating_flag:
            np.random.seed(1000)
            print('regression needs updating')
            df_weather = passed_DataFrame

            unique_period = df_weather.Date.unique()

            test_size = 0.2
            test_choice = np.random.choice([True, False], size=len(unique_period), p=[test_size, 1 - test_size])

            unique_train = unique_period[~test_choice]
            unique_test = unique_period[test_choice]

            df_train = df_weather[df_weather.Date.isin(unique_train)]
            df_test = df_weather[df_weather.Date.isin(unique_test)]

            df_train_X = df_train.drop(['Date', 'WindChillC', 'humidity', 'pressure', 'temperature', 'time', 'wind_speed'],axis=1)
            df_train_Y = df_train['temperature']

            df_test_X = df_test.drop(['Date', 'WindChillC', 'humidity', 'pressure', 'temperature', 'time', 'wind_speed'],axis=1)
            df_test_Y = df_test[['Date', 'time', 'temperature']]

            fitted_reg = reg.fit(df_train_X, df_train_Y)

            pickle_reg = open(file_reg_path, 'wb')
            pickle.dump(fitted_reg, pickle_reg)
            pickle_reg.close()

            return fitted_reg, df_train_X, df_train_Y, df_test_X, df_test_Y

        else:
            print('regression file is present, returning file now')

            regression_in = open(file_reg_path, 'rb')
            fitted_reg = pickle.load(regression_in)

            np.random.seed(1000)
            df_weather = passed_DataFrame
            unique_period = df_weather.Date.unique()

            test_size = 0.2
            test_choice = np.random.choice([True, False], size=len(unique_period), p=[test_size, 1 - test_size])
            unique_train = unique_period[~test_choice]
            unique_test = unique_period[test_choice]

            df_train = df_weather[df_weather.Date.isin(unique_train)]
            df_test = df_weather[df_weather.Date.isin(unique_test)]

            df_train_X = df_train.drop(['Date', 'WindChillC', 'humidity', 'pressure', 'temperature', 'time', 'wind_speed'], axis=1)
            df_train_Y = df_train['temperature']

            df_test_X = df_test.drop(['Date', 'WindChillC', 'humidity', 'pressure', 'temperature', 'time', 'wind_speed'], axis=1)
            df_test_Y = df_test[['Date', 'time', 'temperature']]

            return fitted_reg, df_train_X, df_train_Y, df_test_X, df_test_Y

    else:
        print('creating regression')
        np.random.seed(1000)
        df_weather = passed_DataFrame

        unique_period = df_weather.Date.unique()

        test_size = 0.2
        test_choice = np.random.choice([True, False], size=len(unique_period), p=[test_size, 1 - test_size])

        unique_train = unique_period[~test_choice]
        unique_test = unique_period[test_choice]

        df_train = df_weather[df_weather.Date.isin(unique_train)]
        df_test = df_weather[df_weather.Date.isin(unique_test)]

        df_train_X = df_train.drop(['Date', 'WindChillC', 'humidity', 'pressure', 'temperature', 'time', 'wind_speed'],
                                   axis=1)
        df_train_Y = df_train['temperature']

        df_test_X = df_test.drop(['Date', 'WindChillC', 'humidity', 'pressure', 'temperature', 'time', 'wind_speed'],
                                 axis=1)
        df_test_Y = df_test[['Date', 'time', 'temperature']]

        fitted_reg = reg.fit(df_train_X, df_train_Y)

        pickle_reg = open(file_reg_path, 'wb')
        pickle.dump(fitted_reg, pickle_reg)
        pickle_reg.close()


def graphing(reg, train_X,train_Y, test_X, test_Y):


if __name__ == '__main__':
    def main():
        df_weather, needs_updating_flag = checking_weather_data()

        checking_regression(needs_updating_flag,df_weather)
    main()






'''

def weather_data_checker():
    if weather_data_path.is_file():
        print('weather Data is present')
        weather_data = 'Data//df_weather.p'
        pickle_in = open(weather_data,'rb')
        df_weather = pickle.load(pickle_in)

        df_weather.dropna(inplace=True)

        unique_period = df_weather.Date.unique()

        test_size = 0.2
        test_choice = np.random.choice([True,False], size=len(unique_period), p= [test_size, 1 - test_size])

        unique_train = unique_period[~test_choice]
        unique_test = unique_period[test_choice]

        df_train = df_weather[df_weather.Date.isin(unique_train)]
        df_test = df_weather[df_weather.Date.isin(unique_test)]

        df_train_X = df_train.drop(['Date', 'WindChillC', 'humidity', 'pressure', 'temperature', 'time','wind_speed'],axis = 1)
        df_train_Y = df_train['temperature']

        df_test_X = df_test.drop(['Date', 'WindChillC', 'humidity', 'pressure', 'temperature', 'time','wind_speed'],axis = 1)
        df_test_Y = df_test[['Date','time','temperature']]

    else:
        from Feature_generator import feature_gen
        feature_gen()

        print('weather Data is present')
        weather_data = 'Data//df_weather.p'
        pickle_in = open(weather_data, 'rb')
        df_weather = pickle.load(pickle_in)

        df_weather.dropna(inplace=True)

        unique_period = df_weather.Date.unique()

        test_size = 0.2
        test_choice = np.random.choice([True, False], size=len(unique_period), p=[test_size, 1 - test_size])

        unique_train = unique_period[~test_choice]
        unique_test = unique_period[test_choice]

        df_train = df_weather[df_weather.Date.isin(unique_train)]
        df_test = df_weather[df_weather.Date.isin(unique_test)]

        df_train_X = df_train.drop(['Date', 'WindChillC', 'humidity', 'pressure', 'temperature', 'time', 'wind_speed'],
                                   axis=1)
        df_train_Y = df_train['temperature']

        df_test_X = df_test.drop(['Date', 'WindChillC', 'humidity', 'pressure', 'temperature', 'time', 'wind_speed'],
                                 axis=1)
        df_test_Y = df_test[['Date', 'time', 'temperature']]

    return df_train_X, df_train_Y, df_test_X, df_train_Y

def regression_file_checker():
    df_train_X, df_train_Y, df_test_X,df_test_Y=  weather_data_checker()

    if file_reg_path.is_file() == False:
        print('fitted regression is absent: creating fitting file. \n')

        print('fitting regression ')
        fitted_reg = reg.fit(df_train_X, df_train_Y)

        #### picking the fitted_regression so they You dont have to train over and over ####
        file_reg = 'Data//regression.p'
        pickle_reg = open(file_reg, 'wb')
        pickle.dump(fitted_reg,pickle_reg)
        pickle_reg.close()

        df_test_Y['pred'] = fitted_reg.predict(df_test_X)
        print(df_test_Y.head(100))

        abs_erre = np.round(((df_test_Y.temperature - df_test_Y.pred).abs().sum()/(df_test_Y.temperature).abs().sum())*100,2)
        print(abs_erre)

        r2_error = np.round(r2_score(df_test_Y.temperature, df_test_Y.pred)*100,2)
        print(r2_error)
    else:
        #open reg file
        print('fitted regression is present: loading it in')
        file_reg = 'Data//regression.p'
        regression_in = open(file_reg, 'rb')

        fitted_reg = pickle.load(regression_in)

        df_test_Y['pred'] = fitted_reg.predict(df_test_X)
        print(df_test_Y.head(100))

        abs_erre = np.round(
            ((df_test_Y.temperature - df_test_Y.pred).abs().sum() / (df_test_Y.temperature).abs().sum()) * 100, 2)
        print('Absolute Error:',abs_erre)

        r2_error = np.round(r2_score(df_test_Y.temperature, df_test_Y.pred) * 100, 2)
        print('R2 error:', r2_error)

        date_list = df_test_Y.Date.unique().tolist()
        date = date_list[-1]
        plt.plot(df_test_Y[df_test_Y.Date == date].time, df_test_Y[df_test_Y.Date == date].temperature)
        plt.plot(df_test_Y[df_test_Y.Date == date].time, df_test_Y[df_test_Y.Date == date].pred)
        plt.show()
'''