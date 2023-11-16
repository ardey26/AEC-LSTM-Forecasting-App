import os
import sys

from joblib import load

import numpy as np
import pandas as pd

import tensorflow as tf

import requests
import datetime
from dateutil import relativedelta
import pandas as pd




def load_data(months=1, from_date=str(datetime.date.today())):
    FUTURE_MONTHS = months
    this_dir = os.path.dirname(__file__)
    sys.path.append(os.path.join(this_dir, "datasets/gen_univariate.csv"))
    gen = pd.read_csv(sys.path[-1])
    gen_df = gen.copy()
    gen_df['time'] = pd.to_datetime(gen_df['time'])
    gen_df.set_index('time', inplace=True)
    gen_df = gen_df.loc[:from_date]
    current_gen = gen_df.iloc[-1]


    today = datetime.date.today()
    last_recorded_gen_date = datetime.date(pd.to_datetime(gen.iloc[-1].time).year, pd.to_datetime(gen.iloc[-1].time).month, pd.to_datetime(gen.iloc[-1].time).day)
    if today > last_recorded_gen_date:
        today = last_recorded_gen_date
    print(pd.to_datetime(from_date),'aopwidjawiodawiopjda')
    last_month = datetime.date(pd.to_datetime(from_date).year, pd.to_datetime(from_date).month, pd.to_datetime(from_date).day) - relativedelta.relativedelta(months=FUTURE_MONTHS)

    response = requests.get(f'https://archive-api.open-meteo.com/v1/archive?latitude=15.15&longitude=120.5833&start_date={last_month}&end_date={from_date}&daily=temperature_2m_max&timezone=auto')
    
    if response:
        temps = response.json()['daily']['temperature_2m_max']
        time = response.json()['daily']['time']
    temps_df = pd.DataFrame({"T2_MAX": temps}, index=pd.to_datetime(time))
    temps_df = temps_df.resample("m").mean()
    concat_df = pd.concat((gen_df, temps_df), axis=1)
    concat_df = concat_df.iloc[-(FUTURE_MONTHS + 1):]
    print(concat_df)
    return concat_df, current_gen.values[0]

def load_all_temps(from_date=None):
    today = datetime.date.today()
    print(today)
    response = requests.get(f'https://archive-api.open-meteo.com/v1/archive?latitude=15.15&longitude=120.5833&start_date=2008-07-01&end_date={today}&daily=temperature_2m_max&timezone=auto')
    
    if response:
        temps = response.json()['daily']['temperature_2m_max']
        time = response.json()['daily']['time']
    temps_df = pd.DataFrame({"T2_MAX": temps}, index=pd.to_datetime(time))
    temps_df = temps_df.resample("m").mean()
    if from_date:
        temps, dates = temps_df.loc[from_date:, 'T2_MAX'].values.tolist(), temps_df.loc[from_date:, 'T2_MAX'].index.strftime("%Y-%m-%d").tolist()[:-1]    
    else:
        temps, dates = temps_df.iloc[:, 0].values.tolist(), temps_df.index.strftime("%Y-%m-%d").tolist()[:-1]
    return temps, dates

def load_all_gen(from_date=None):
    this_dir = os.path.dirname(__file__)
    sys.path.append(os.path.join(this_dir, "datasets/gen_univariate.csv"))
    gen = pd.read_csv(sys.path[-1])
    
    gen_df = gen.copy()
    gen_df['time'] = pd.to_datetime(gen_df['time'])
    gen_df.set_index('time', inplace=True)
    if from_date:
        gen = gen_df.loc[from_date:, 'gen'].values.tolist()
    else:
        gen = gen_df.iloc[:, 0].values.tolist()
        
    return gen
# def load_current_gen():	
#     storage_options = {'User-Agent': 'Mozilla/5.0'}
#     file_id = '1gPBGi203ReKeUq4Hqrfc-Mut7hdPqOBq'
#     url = 'https://drive.google.com/uc?id={}'.format(file_id)
#     gen = pd.read_csv(url, storage_options=storage_options)
#     gen_df = gen.copy()
#     gen_df['time'] = pd.to_datetime(gen_df['time'])
#     gen_df.set_index('time', inplace=True)
#     gen_df = gen_df.iloc[-1:]
#     curr_gen = gen_df['gen'].values[0]
#     return curr_gen

    

def load_model(file_name):
    model = tf.keras.models.load_model(file_name)
    return model

def load_scaler(file_name):
    this_dir = os.path.dirname(__file__)
    sys.path.append(os.path.join(this_dir, file_name))
    scaler = load(sys.path[-1])
    return scaler

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def preprocess_to_prediction_sequence(dataset, model, scaler, months=1):
  reframed = series_to_supervised(dataset, months, 1)

  reframed.drop(reframed.columns[-1], axis=1, inplace=True)
  values = scaler.transform(reframed)

  X = values[:, :-1]
  y = values[:, -1]

  X = X.reshape((X.shape[0], 1, X.shape[1]))

  y_pred = model.predict(X)

  X = X.reshape((X.shape[0], X.shape[2]))
  pred_concat = np.concatenate((y_pred, X), axis=1)
  pred_concat = scaler.inverse_transform(pred_concat)
  pred_concat = pred_concat[:, 0]

  return pred_concat.tolist()

