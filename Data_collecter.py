import pandas as pd
import json
from datetime import timedelta
from urllib.request import urlopen

def checker():
    pass

def weather_history(date_go_back = '2009-01-01', postcode_sel = 'EC1A 1AA', API_key =  'def583708ca749ef8cc210541172408',):
    now = pd.datetime.today()                   # gets a date for the current day and converst it
    now = now.strftime('%Y-%m-%d')
    now = str(now)

    date = pd.date_range( date_go_back, now, freq='MS')  # makes a list of date ranges used for collecting
    date = date.strftime('%Y-%m-%d')
    date_list = []
    for head_date in date:
        date_list.append(str(head_date))

    date_list.append(now)
    print(date_list)

    post_code = postcode_sel
    url_post = 'http://api.postcodes.io/postcodes/' + post_code
    obj_post = urlopen(url_post)
    post_data = json.load(obj_post)
    print(post_data['result']['parliamentary_constituency'])

    df_weather = []
    for iter in range(1,len(date_list)):
        print(iter, 'start', date_list[iter], '\t end', date_list[iter - 1])
        key = API_key
        url_weather = 'http://api.worldweatheronline.com/premium/v1/past-weather.ashx?key=' + key + '&q=london&format=json&date=' + date_list[iter-1] + '&enddate=' + date_list[iter]+ '&tp=1'
        print(url_weather)
        obj_weather = urlopen(url_weather)
        weather_data = json.load(obj_weather)

        for head_date in range(len(weather_data['data']['weather'])):
            date = weather_data['data']['weather'][head_date]['date']
            #print(date)

            for Tail_date in range(len(weather_data['data']['weather'][0]['hourly'])):
                time = weather_data['data']['weather'][head_date]['hourly'][Tail_date]['time']
                tempc = weather_data['data']['weather'][head_date]['hourly'][Tail_date]['tempC']
                wind_speed = weather_data['data']['weather'][head_date]['hourly'][Tail_date]['windspeedKmph']
                humidity = weather_data['data']['weather'][head_date]['hourly'][Tail_date]['humidity']
                pressure =  weather_data['data']['weather'][head_date]['hourly'][Tail_date]['pressure']
                WindChillC = weather_data['data']['weather'][head_date]['hourly'][Tail_date]['WindChillC']

                df_weather.append({'Date':date,'time':time,'temperature':tempc,'wind_speed':wind_speed,'humidity':humidity,'pressure':pressure,'WindChillC':WindChillC})

           # print('Date',date, 'Time',time, 'tempC',tempc, 'wind_speed',wind_speed,
                  #'humidity',humidity, 'pressure',pressure, 'WindChillC',WindChillC)

    df_weather = pd.DataFrame(df_weather)
    df_weather.drop_duplicates(inplace=True)
    return df_weather


