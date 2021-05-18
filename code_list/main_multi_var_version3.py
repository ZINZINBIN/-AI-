# ======================================================================== #
# ========================== From gitlab Clone =========================== #
# ======================================================================== #

import os
import numpy as np
import pandas as pd
import scipy as sp
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import glob
from lib_function import *

# path
path = "original_dataset/"
dangjin_fcst_data_path = path + "dangjin_fcst_data.csv"
dangjin_obs_data_path = path + "dangjin_obs_data.csv"
energy_data_path = path + "energy.csv"
ulsan_fcst_data_path = path + "ulsan_fcst_data.csv"
ulsan_obs_data_path = path + "ulsan_obs_data.csv"
site_info_path = path + "site_info.csv"

# file convert to pandas data
dangjin_fcst_data = pd.read_csv(dangjin_fcst_data_path)
dangjin_obs_data = pd.read_csv(dangjin_obs_data_path)
energy_data = pd.read_csv(energy_data_path)
ulsan_fcst_data = pd.read_csv(ulsan_fcst_data_path)
ulsan_obs_data = pd.read_csv(ulsan_obs_data_path)
site_info = pd.read_csv(site_info_path)


dangjin_fcst_data.rename(
    columns = {
        "Forecast time":"time", 
        "forecast":"forecast_fcst", 
        "Temperature":"temp_fcst",
        "Humidity":"humid_fcst",
        "WindSpeed":"ws_fcst",
        "WindDirection":"wd_fcst",
        "Cloud":"cloud_fcst"
        }, inplace = True)

dangjin_obs_data.rename(
    columns = {
        "일시":"time",
        "기온(°C)":"temp_obs",
        "풍속(m/s)":"ws_obs",
        "풍향(16방위)":"wd_obs",
        "습도(%)":"humid_obs",
        "전운량(10분위)":"cloud_obs"
    }, inplace = True)

ulsan_fcst_data.rename(
    columns = {
        "Forecast time":"time", 
        "forecast":"forecast_fcst", 
        "Temperature":"temp_fcst",
        "Humidity":"humid_fcst",
        "WindSpeed":"ws_fcst",
        "WindDirection":"wd_fcst",
        "Cloud":"cloud_fcst"
    }, inplace = True)

ulsan_obs_data.rename(
    columns = {
        "일시":"time",
        "기온(°C)":"temp_obs",
        "풍속(m/s)":"ws_obs",
        "풍향(16방위)":"wd_obs",
        "습도(%)":"humid_obs",
        "전운량(10분위)":"cloud_obs"
    }, inplace = True)

dangjin_obs_data = dangjin_obs_data.drop(columns = ["지점", "지점명"])
ulsan_obs_data = ulsan_obs_data.drop(columns = ["지점","지점명"])

# fcst_data 데이터 전처리
# time + forecast -> time으로 전환, 이후 중복되는 값은 평균 처리 

dangjin_fcst_data["time_fcst"] = pd.to_datetime(dangjin_fcst_data["time"].copy()) + dangjin_fcst_data["forecast_fcst"].copy().astype("timedelta64[h]")
#dangjin_fcst_data = dangjin_fcst_data.groupby("time_fcst", as_index = False).mean()
dangjin_fcst_data = dangjin_fcst_data.groupby("time_fcst", as_index = False).last()

dangjin_fcst_data = dangjin_fcst_data.drop(columns = ["forecast_fcst", "time"])

ulsan_fcst_data["time_fcst"] = pd.to_datetime(ulsan_fcst_data["time"].copy()) + ulsan_fcst_data["forecast_fcst"].copy().astype("timedelta64[h]")
#ulsan_fcst_data = ulsan_fcst_data.groupby("time_fcst", as_index = False).mean()
ulsan_fcst_data = ulsan_fcst_data.groupby("time_fcst", as_index = False).last()
ulsan_fcst_data = ulsan_fcst_data.drop(columns = ["forecast_fcst", "time"])

ulsan_fcst_data.rename(columns = {"time_fcst":"time"}, inplace = True)
dangjin_fcst_data.rename(columns = {"time_fcst":"time"}, inplace = True)

ulsan_fcst_data = ulsan_fcst_data.astype({"time":"object"})
dangjin_fcst_data = dangjin_fcst_data.astype({"time":"object"})

dangjin_obs_data["time"] = pd.to_datetime(dangjin_obs_data["time"].copy(), format='%Y-%m-%d %H:%M:%S')
dangjin_obs_data = dangjin_obs_data.astype({"time":"object"})

ulsan_obs_data["time"] = pd.to_datetime(ulsan_obs_data["time"].copy(), format='%Y-%m-%d %H:%M:%S')
ulsan_obs_data = ulsan_obs_data.astype({"time":"object"})

# energy_data는 time 항목이 string으로 저장되어 있다. 이를 timestamp로 처리해야한다. 

import datetime as dt

energy_data_time_tmp = energy_data["time"].copy()

for i in range(energy_data.shape[0]):
    if energy_data["time"][i][-8:] == "24:00:00":
        energy_data["time"][i] = energy_data_time_tmp[i].replace("24:00:00", " 00:00:00")
        energy_data["time"][i] = pd.to_datetime(energy_data["time"][i]) + dt.timedelta(days = 1)

    energy_data["time"][i] = pd.Timestamp(energy_data["time"][i])

energy_data = energy_data.astype({"time":"object"})

# 전 데이터 NAN 처리

dangjin_fcst_data = dangjin_fcst_data.fillna(method = "bfill")
dangjin_obs_data = dangjin_obs_data.fillna(method = "bfill")
energy_data = energy_data.fillna(method = "bfill")
ulsan_fcst_data = ulsan_fcst_data.fillna(method = "bfill")
ulsan_obs_data = ulsan_obs_data.fillna(method = "bfill")

# fcst_data['time'] time interval: 3hour -> 1hour로 축소 필요
# Lagrangian Interpolation

def interpolation(df):

    df_copy = df.copy()
    var_names = df.columns

    total_s = list()
    time_list = list()
    
    for var_name in var_names:
        s = list()
        for i in range(df_copy.shape[0] - 1):
            timedeltas = df_copy["time"][i+1] - df_copy["time"][i]
            n_intervals = int(timedeltas / np.timedelta64(1, "h"))

            for j in range(n_intervals):
        
                if var_name == "time":
                    time_stamps = df_copy["time"][i] + timedeltas * j / n_intervals
                    time_list.append(time_stamps)
                else:
                    add_ = df_copy[var_name][i] + (df_copy[var_name][i+1] - df_copy[var_name][i]) / n_intervals * j
                    s.append(add_)

        if var_name == "time":
            time_list = np.array(time_list).reshape(-1,1)
            total_s.append(time_list)
        else:
            s = np.array(s).reshape(-1,1)
            total_s.append(s)

    total_s = np.array(total_s).T.reshape(-1, len(var_names))
    df_converted = pd.DataFrame(total_s, columns = var_names)

    return df_converted

dangjin_fcst_data = interpolation(dangjin_fcst_data.copy())
ulsan_fcst_data = interpolation(ulsan_fcst_data.copy())

ulsan_fcst_data = ulsan_fcst_data.astype({"time":"object"})
dangjin_fcst_data = dangjin_fcst_data.astype({"time":"object"})
energy_data = energy_data.astype({"time":"object"})
dangjin_obs_data = dangjin_obs_data.astype({"time":"object"})
ulsan_obs_data = ulsan_obs_data.astype({"time":"object"})

# total dataset 구성

from functools import reduce

list_dangjin = [dangjin_fcst_data, dangjin_obs_data, energy_data[["time","dangjin_floating","dangjin_warehouse","dangjin"]].copy()]
list_ulsan = [ulsan_fcst_data, ulsan_obs_data, energy_data[["time","ulsan"]].copy()]

dangjin_data = reduce(lambda  left,right: pd.merge(left, right, on=['time'], how='inner'), list_dangjin)
ulsan_data = reduce(lambda  left,right: pd.merge(left, right, on=['time'], how='inner'), list_ulsan)

# data의 month,day, time을 추가하자

month = []
day = []
hour = []

for i in range(len(dangjin_data)):
    month.append(dangjin_data["time"][i].month)
    day.append(dangjin_data["time"][i].day)
    hour.append(dangjin_data["time"][i].hour)

month = np.array(month).reshape(-1,1)
day = np.array(day).reshape(-1,1)
hour = np.array(hour).reshape(-1,1)

dangjin_data["month"] = month
dangjin_data["day"] = day
dangjin_data["hour"] = hour


month = []
day = []
hour = []

for i in range(len(ulsan_data)):
    month.append(ulsan_data["time"][i].month)
    day.append(ulsan_data["time"][i].day)
    hour.append(ulsan_data["time"][i].hour)

month = np.array(month).reshape(-1,1)
day = np.array(day).reshape(-1,1)
hour = np.array(hour).reshape(-1,1)

ulsan_data["month"] = month
ulsan_data["day"] = day
ulsan_data["hour"] = hour


'''
# preprocessing
from sklearn.preprocessing import MinMaxScaler

def preprocessing(data, col_name):
    # col_name: column name(string type)
    # data: pd.DataFrame
    
    data_np = data[col_name].values
    
    scaler = MinMaxScaler()
    data_np = scaler.fit_transform(data_np).reshape(-1, len(col_name))
    
    data = data.drop(columns = col_name)
    data[col_name] = data_np
    
    return data, scaler
    
def PG_preprocessing(data, name = "ulsan"):
    data_np = data[name].values.reshape(-1,1)
    
    scaler = MinMaxScaler()
    
    data_np = scaler.fit_transform(data_np).reshape(-1,1)
    
    data = data.drop(columns = name)
    data[name] = data_np
    
    return data, scaler

def PG_inverse(y, scaler):
    # ulsan, dangjin_floating, warehouse, dangjin data scaler
    # inverse transform to its own unit
    
    if y.shape[1] != 1:
        y = y.reshape(-1,1)
    y_rs = scaler.inverse_transform(y).reshape(-1,1)
    return y_rs

#col_name = ["temp_fcst", "temp_obs", "ws_fcst", "ws_obs", "humid_fcst", "humid_obs"]

col_name = ["temp_obs", "ws_obs", "humid_obs"]
ulsan_data, ulsan_scaler_mv = preprocessing(ulsan_data, col_name)
dangjin_data, dangjin_scaler_mv = preprocessing(dangjin_data, col_name)

ulsan_data, ulsan_scaler = PG_preprocessing(ulsan_data, "ulsan")

dangjin_data, dangjin_floating_scaler = PG_preprocessing(dangjin_data, "dangjin_floating")
dangjin_data, dangjin_warehouse_scaler = PG_preprocessing(dangjin_data, "dangjin_warehouse")
dangjin_data, dangjin_scaler = PG_preprocessing(dangjin_data, "dangjin")

'''

# 예측에 활용할 fcst 데이터는 0을 소거 x
ulsan_data_for_predict = ulsan_data.copy()
dangjin_data_for_predict = dangjin_data.copy()


'''
# 0도 소거
dangjin_data = dangjin_data[dangjin_data["dangjin"] != 0]
dangjin_data = dangjin_data[dangjin_data["dangjin_warehouse"] != 0]
dangjin_data = dangjin_data[dangjin_data["dangjin_floating"] != 0]

ulsan_data = ulsan_data[ulsan_data["ulsan"] != 0]

dangjin_data.reset_index(drop = True, inplace = True)
ulsan_data.reset_index(drop = True, inplace = True)

'''

# total dataset summary
display(dangjin_data)
display(ulsan_data)


'''
- data_generator 함수를 변경할 예정
- month, day, hour는 그대로, 그 이후 timesteps에 따른 변수명을 달리할 예정
'''

def series_to_supervised2(data, x_name, y_name, n_in, n_out, dropnan = False):

    '''
    - function: to convert series data to be supervised 
    - data: pd.DataFrame
    - x_name: the name of variables used to predict
    - y_name: the name of variables for prediction
    - n_in: number(or interval) of series used to predict
    - n_out: number of series for prediction

    - 24 * 30 -> 720개의 output을 예측
    - 필요한 input -> 최소 720개 이상
    - 아이디어: 1일 예측, 예측치를 다시 입력값으로 받게 진행, 이 경우 output:24

    '''

    data_copy = data.copy()
    cols, names = list(), list()


    for i in range(n_in, 0, -1):
        cols.append(data_copy[x_name].shift(i))
        names += [("%s(t-%d)"%(name, i)) for name in x_name]
    
    cols.append(data_copy["month"])
    cols.append(data_copy["day"])
    cols.append(data_copy["hour"])
    
    names += ["month"]
    names += ["day"]
    names += ["hour"]
    
    for i in range(0, n_out):
        y = data_copy[y_name]
        cols.append(y.shift(-i))
        # cols:[data_copy.shift(n_in-1), .... data_copy.shift(1), data_copy[y_name].shift(0)....data_copy[y_name].shift(-n_out + 1)]

        if i == 0:
            names += [("%s(t)"%(name)) for name in y_name]
        else:
            names += [("%s(t+%d)"%(name, i)) for name in y_name]

    agg = pd.concat(cols, axis = 1)
    agg.columns = names

    if dropnan:
        agg.dropna(inplace = True)
    
    return agg

def data_generator2(data, n_in, n_out, ratio, x_name, y_name):
    data_supervised = series_to_supervised2(data, x_name, y_name, n_in, n_out, dropnan = True)
    
    x_data = data_supervised.values[:, :-n_out * len(y_name)]
    y_data = data_supervised.values[:, -n_out * len(y_name):]

    data_size = x_data.shape[0]
    train_size = int(data_size * ratio)
    
    x_train = x_data[0:train_size]
    x_test = x_data[train_size:]

    y_train = y_data[0:train_size]
    y_test = y_data[train_size:]

    return (x_train, y_train), (x_test, y_test)


# model architecture
import xgboost as xgb

params = {
    "booster":"dart",
    "objective":"reg:pseudohubererror",
    #"objective":"reg:squarederror",
    "learning_rate":0.2,
    #"max_depth":7,
    "max_depth":11,
    "n_estimators":1024 * 2,
    "nthread":-1,
    "gamma":1.0,
    "subsample":0.7,
    "colsample_bytree":1.0,
    "colsample_bylevel":1.0,
    "min_child_weight":5, #5
    "reg_lambda":0.1,
    "reg_alpha":1.0, # 1.0
    "sample_type":"weighted"
}


'''
# custom objective function application
def custom_objective(labels, preds):
    grad = 
    hess = 
'''

capacity = {
    'dangjin_floating':1000, # 당진수상태양광 발전용량
    'dangjin_warehouse':700, # 당진자재창고태양광 발전용량
    'dangjin':1000, # 당진태양광 발전용량
    'ulsan':500 # 울산태양광 발전용량
}

def custom_evaluation(preds, dtrain, cap = "ulsan"):
    labels = dtrain.get_label()
    abs_err = np.absolute(labels - preds)
    abs_err /= capacity[cap]
    target_idx = np.where(labels >= capacity[cap] * 0.1)
    result = 100 * abs_err[target_idx].mean()
    return "eval_NMAE", result


n_out = 1
ratio = 0.8 # train size ratio
n_in_list = [8, 24, 24 * 3, 24 * 7, 24 * 15, 24 * 30]
esr = 400 # early stopping round

# ulsan model
x_name = ["temp_obs","ws_obs","humid_obs","ulsan"]
y_name = ["ulsan"]

n_features = len(x_name)
n_out = 1
ratio = 0.8 # train size ratio

# preprocessing
# scaling

ulsan_models = [build_xgb(params) for n_in in n_in_list]

custom_eval_ulsan = lambda x,y : custom_evaluation(x, y, cap = "ulsan")

for model, n_in in zip(ulsan_models, n_in_list):
    (x_train, y_train), (x_test, y_test) = data_generator2(ulsan_data, n_in, n_out, ratio, x_name, y_name)
    model.fit(x_train, y_train, eval_set = [(x_test, y_test)], early_stopping_rounds = esr, eval_metric = custom_eval_ulsan)
    
    del x_train, x_test, y_train, y_test

# dangjin_floating_models training
x_name = ["temp_obs","ws_obs","humid_obs","dangjin_floating"]
y_name = ["dangjin_floating"]

dangjin_floating_models = [build_xgb(params) for n_in in n_in_list]

custom_eval_dangjin_floating = lambda x,y : custom_evaluation(x, y, cap = "dangjin_floating")

for model, n_in in zip(dangjin_floating_models, n_in_list):
    (x_train, y_train), (x_test, y_test) = data_generator2(dangjin_data, n_in, n_out, ratio, x_name, y_name)
    model.fit(x_train, y_train, eval_set = [(x_test, y_test)], early_stopping_rounds = esr, eval_metric = custom_eval_dangjin_floating)
    del x_train, x_test, y_train, y_test


# dangjin_warehouse_models training
x_name = ["temp_obs","ws_obs","humid_obs","dangjin_warehouse"]
y_name = ["dangjin_warehouse"]
dangjin_warehouse_models = [build_xgb(params) for n_in in n_in_list]

custom_eval_dangjin_warehouse = lambda x,y : custom_evaluation(x, y, cap = "dangjin_warehouse")

for model, n_in in zip(dangjin_warehouse_models, n_in_list):
    (x_train, y_train), (x_test, y_test) = data_generator2(dangjin_data, n_in, n_out, ratio, x_name, y_name)
    model.fit(x_train, y_train, eval_set = [(x_test, y_test)], early_stopping_rounds = esr, eval_metric = custom_eval_dangjin_warehouse)
    del x_train, x_test, y_train, y_test

# dangjin_models training
x_name = ["temp_obs","ws_obs","humid_obs","dangjin"]
y_name = ["dangjin"]
dangjin_models = [build_xgb(params) for n_in in n_in_list]

custom_eval_dangjin = lambda x,y : custom_evaluation(x, y, cap = "dangjin")

for model, n_in in zip(dangjin_models, n_in_list):
    (x_train, y_train), (x_test, y_test) = data_generator2(dangjin_data, n_in, n_out, ratio, x_name, y_name)
    model.fit(x_train, y_train, eval_set = [(x_test, y_test)], early_stopping_rounds = esr, eval_metric = custom_eval_dangjin)
    del x_train, x_test, y_train, y_test

# forecasting
# 21.01.01 - 21.01.31
term_3d = range(0, 24 * 3)
term_7d = range(0, 24 * 7)
term_30d = range(0, 24 * 30)

# comparing result
def submission_predict2(model, x_data, n_predict, model_type = "uv", x_date = None):

    '''
    - model_type: "mv" or "uv"
    (1) model_type = "uv"
        - forecasting uni-variable for n_predict timesteps
    (2) model_type = "mv"
        - forecasting with multi variables for n_predict timesteps
        - model input: (timesteps, n_features{temp, humid, ws, power})
        - x_data : (obs_data, fcst_data), obs_data[t - timesteps,...., t-1], fcst_data[t,....,t+1month]
        - obs_data.shape: timesteps * n_features
        - fcst_data.shape: n_predict * (n_features - 1)
    - x_data:(timesteps, n_features)
    - n_predict: timesteps for forecasting
    - x_data_after: x_data[1:] + predict_value
    '''
    
    total_prediction = None
    y_preds = []
    

    month, day, hour = x_date

    if model_type == "uv":
        x_data_after = x_data
        for i in range(n_predict):
            
            add_date = np.array([month[i], day[i], hour[i]]).reshape(1, -1)
            input_data = np.concatenate((x_data_after, add_date), axis = 1)
            
            y_pred = model.predict(input_data)
            x_data_after = np.append(x_data_after, y_pred)[1:].reshape(1,-1)
            y_preds.append(y_pred)
        
        total_prediction = np.array(y_preds).reshape(-1,1)

    elif model_type == "mv":
        # obs_data: 1월 31일 이전 데이터(관측값)
        # fcst_dat: 2월 1일 이후 데이터(예측값)
        obs_data, fcst_data = x_data
        x_data_after = obs_data
        nf = fcst_data.shape[1] + 1
        for i in range(n_predict):

            add_date = np.array([month[i], day[i], hour[i]]).reshape(1, -1)
            #print(add_date.shape)
            input_data = np.concatenate((x_data_after, add_date), axis = 1)
            #print(input_data.shape)
            y_pred = model.predict(input_data) # y_pred는 낱개의 예측치
            #print(y_pred)
            add_data = np.append(fcst_data[i,:].reshape(1,-1), y_pred).reshape(1,-1)
            #add_data = np.concatenate((fcst_data[i,:].reshape(1,-1), y_pred), axis = 1)

            x_data_after = np.concatenate((x_data_after, add_data), axis = 1)[0, nf:].reshape(1,-1)
            y_preds.append(y_pred)
            #print(x_data_after)

        total_prediction = np.array(y_preds).reshape(-1,1)

    return total_prediction


# ulsan prediction
x_name = ["temp_obs","ws_obs","humid_obs","ulsan"]
y_name = ["ulsan"]
yhats_ulsan = None

m = ulsan_data["month"][- 24*30*1:].values.reshape(-1,1)
d = ulsan_data["day"][- 24*30*1:].values.reshape(-1,1)
h = ulsan_data["hour"][- 24*30*1:].values.reshape(-1,1)
    
d_date = (m,d,h)

for n_in, model in zip(n_in_list, ulsan_models):
    name = "ulsan, timesteps: " + str(n_in / 24) + "-day"
    d_obs = ulsan_data_for_predict[x_name][-24*30*1 - n_in : - 24*30*1].values.reshape(1,-1)
    d_fcst = ulsan_data_for_predict[x_name[0:-1]][-24*30*1:].values.reshape(24*30, len(x_name[0:-1]))
    prediction = submission_predict2(model, (d_obs, d_fcst), n_predict = 24 * 30, model_type = "mv",  x_date = d_date)

    actual = ulsan_data_for_predict[y_name][-24*30*1:].values
    yreal = actual.reshape(-1,1)
    #yreal = PG_inverse(yreal, ulsan_scaler)
    
    yhat = prediction.reshape(-1,1)
    #yhat = PG_inverse(yhat, ulsan_scaler)

    if yhats_ulsan is None:
        yhats_ulsan = yhat
    else:
        yhats_ulsan = np.concatenate((yhats_ulsan, yhat), axis = 1)

    label = "forecast"

    for i, term in enumerate([term_3d, term_7d, term_30d]):
        plt.figure(i+1, figsize = (10, 5))
        plt.plot(yreal[term], label = "real")
        plt.plot(yhat[term], label = label)
        plt.ylabel("ulsan, unit:None")
        plt.title(name)
        plt.legend()
        plt.show()

# dangjin_floating prediction
x_name = ["temp_obs","ws_obs","humid_obs","dangjin_floating"]
y_name = ["dangjin_floating"]
yhats_dangjin_floating = None

m = dangjin_data["month"][- 24*30*1:].values.reshape(-1,1)
d = dangjin_data["day"][- 24*30*1:].values.reshape(-1,1)
h = dangjin_data["hour"][- 24*30*1:].values.reshape(-1,1)
    
d_date = (m,d,h)

for n_in, model in zip(n_in_list, dangjin_floating_models):
    name = "dangjin_floating, timesteps: " + str(n_in / 24) + "-day"
    d_obs = dangjin_data_for_predict[x_name][-24*30*1 - n_in : - 24*30*1].values.reshape(1,-1)
    d_fcst = dangjin_data_for_predict[x_name[0:-1]][-24*30*1:].values.reshape(24*30, len(x_name[0:-1]))
    prediction = submission_predict2(model, (d_obs, d_fcst), n_predict = 24 * 30, model_type = "mv",  x_date = d_date)

    actual = dangjin_data_for_predict[y_name][-24*30*1:].values
    yreal = actual.reshape(-1,1)
    #yreal = PG_inverse(yreal, dangjin_floating_scaler)
    
    yhat = prediction.reshape(-1,1)
    #yhat = PG_inverse(yhat, dangjin_floating_scaler)

    if yhats_dangjin_floating is None:
        yhats_dangjin_floating = yhat
    else:
        yhats_dangjin_floating = np.concatenate((yhats_dangjin_floating, yhat), axis = 1)

    label = "forecast"

    for i, term in enumerate([term_3d, term_7d, term_30d]):
        plt.figure(i+1, figsize = (10, 5))
        plt.plot(yreal[term], label = "real")
        plt.plot(yhat[term], label = label)
        plt.ylabel("dangjin_floating, unit:None")
        plt.title(name)
        plt.legend()
        plt.show()

# dangjin_warehouse prediction
x_name = ["temp_obs","ws_obs","humid_obs","dangjin_warehouse"]
y_name = ["dangjin_warehouse"]
yhats_dangjin_warehouse = None

m = dangjin_data["month"][- 24*30*1:].values.reshape(-1,1)
d = dangjin_data["day"][- 24*30*1:].values.reshape(-1,1)
h = dangjin_data["hour"][- 24*30*1:].values.reshape(-1,1)
    
d_date = (m,d,h)

for n_in, model in zip(n_in_list, dangjin_warehouse_models):
    name = "dangjin_warehouse, timesteps: " + str(n_in / 24) + "-day"
    d_obs = dangjin_data_for_predict[x_name][-24*30*1 - n_in : - 24*30*1].values.reshape(1,-1)
    d_fcst = dangjin_data_for_predict[x_name[0:-1]][-24*30*1:].values.reshape(24*30, len(x_name[0:-1]))
    prediction = submission_predict2(model, (d_obs, d_fcst), n_predict = 24 * 30, model_type = "mv",  x_date = d_date)

    actual = dangjin_data_for_predict[y_name][-24*30*1:].values
    yreal = actual.reshape(-1,1)
    #yreal = PG_inverse(yreal, dangjin_warehouse_scaler)
    
    yhat = prediction.reshape(-1,1)
    #yhat = PG_inverse(yhat, dangjin_warehouse_scaler)

    if yhats_dangjin_warehouse is None:
        yhats_dangjin_warehouse = yhat
    else:
        yhats_dangjin_warehouse = np.concatenate((yhats_dangjin_warehouse, yhat), axis = 1)

    label = "forecast"

    for i, term in enumerate([term_3d, term_7d, term_30d]):
        plt.figure(i+1, figsize = (10, 5))
        plt.plot(yreal[term], label = "real")
        plt.plot(yhat[term], label = label)
        plt.ylabel("dangjin_warehouse, unit:None")
        plt.title(name)
        plt.legend()
        plt.show()

# dangjin prediction
x_name = ["temp_obs","ws_obs","humid_obs","dangjin"]
y_name = ["dangjin"]
yhats_dangjin = None

m = dangjin_data["month"][- 24*30*1:].values.reshape(-1,1)
d = dangjin_data["day"][- 24*30*1:].values.reshape(-1,1)
h = dangjin_data["hour"][- 24*30*1:].values.reshape(-1,1)
    
d_date = (m,d,h)

for n_in, model in zip(n_in_list, dangjin_models):
    name = "dangjin, timesteps: " + str(n_in / 24) + "-day"
    d_obs = dangjin_data_for_predict[x_name][-24*30*1 - n_in : - 24*30*1].values.reshape(1,-1)
    d_fcst = dangjin_data_for_predict[x_name[0:-1]][-24*30*1:].values.reshape(24*30, len(x_name[0:-1]))
    prediction = submission_predict2(model, (d_obs, d_fcst), n_predict = 24 * 30, model_type = "mv",  x_date = d_date)

    actual = dangjin_data_for_predict[y_name][-24*30*1:].values
    yreal = actual.reshape(-1,1)
    #yreal = PG_inverse(yreal, dangjin_scaler)
    
    yhat = prediction.reshape(-1,1)
    #yhat = PG_inverse(yhat, dangjin_scaler)

    if yhats_dangjin is None:
        yhats_dangjin = yhat
    else:
        yhats_dangjin = np.concatenate((yhats_dangjin, yhat), axis = 1)

    label = "forecast"

    for i, term in enumerate([term_3d, term_7d, term_30d]):
        plt.figure(i+1, figsize = (10, 5))
        plt.plot(yreal[term], label = "real")
        plt.plot(yhat[term], label = label)
        plt.ylabel("dangjin, unit:None")
        plt.title(name)
        plt.legend()
        plt.show()

# ensemble: weighted sum
# ulsan data
model_num = len(ulsan_models)
actual = ulsan_data["ulsan"][-24*30*1:].values
yreal = actual.reshape(-1,1)
#yreal = PG_inverse(yreal, ulsan_scaler)

w_ulsan = ensemble_weights(yhats_ulsan, yreal, model_num)
ulsan_weighted_sum = np.dot(yhats_ulsan, w_ulsan).reshape(-1,1)
#ulsan_weighted_sum = PG_inverse(ulsan_weighted_sum, ulsan_scaler)

term = term_30d
plt.plot(yreal[term], label = "real")
plt.plot(ulsan_weighted_sum[term], label = "weighted sum")
plt.ylabel("ulsan, unit:None")
plt.title("Real and ensemble, ulsan")
plt.legend()
plt.show()

# dangjin data
# dangjin_floating
actual = dangjin_data["dangjin_floating"][-24*30*1:].values
yreal = actual.reshape(-1,1)
#yreal = PG_inverse(yreal, dangjin_floating_scaler)

w_dangjin_floating = ensemble_weights(yhats_dangjin_floating, yreal, model_num)
dangjin_floating_weighted_sum = np.dot(yhats_dangjin_floating, w_dangjin_floating).reshape(-1,1)
#dangjin_floating_weighted_sum = PG_inverse(dangjin_floating_weighted_sum, dangjin_floating_scaler)

term = term_30d
plt.plot(yreal[term], label = "real")
plt.plot(dangjin_floating_weighted_sum[term], label = "weighted sum")
plt.ylabel("dangjin_floating, unit:None")
plt.title("Real and ensemble, dangjin_floating")
plt.legend()
plt.show()

# dangjin_warehouse
actual = dangjin_data["dangjin_warehouse"][-24*30*1:].values
yreal = actual.reshape(-1,1)
#yreal = PG_inverse(yreal, dangjin_warehouse_scaler)

w_dangjin_warehouse = ensemble_weights(yhats_dangjin_warehouse, yreal, model_num)
dangjin_warehouse_weighted_sum = np.dot(yhats_dangjin_warehouse, w_dangjin_warehouse).reshape(-1,1)
#dangjin_warehouse_weighted_sum = PG_inverse(dangjin_warehouse_weighted_sum, dangjin_warehouse_scaler)


term = term_30d
plt.plot(yreal[term], label = "real")
plt.plot(dangjin_warehouse_weighted_sum[term], label = "weighted sum")
plt.ylabel("dangjin_warehouse, unit:None")
plt.title("Real and ensemble, dangjin_warehouse")
plt.legend()
plt.show()

# dangjin
actual = dangjin_data["dangjin"][-24*30*1:].values
yreal = actual.reshape(-1,1)
#yreal = PG_inverse(yreal, dangjin_scaler)

w_dangjin = ensemble_weights(yhats_dangjin, yreal, model_num)
dangjin_weighted_sum = np.dot(yhats_dangjin, w_dangjin).reshape(-1,1)
#dangjin_weighted_sum = PG_inverse(dangjin_weighted_sum, dangjin_scaler)


term = term_30d
plt.plot(yreal[term], label = "real")
plt.plot(dangjin_weighted_sum[term], label = "weighted sum")
plt.ylabel("dangjin, unit:None")
plt.title("Real and ensemble, dangjin")
plt.legend()
plt.show()


# submission
submission_path = path + "sample_submission.csv"
submission = pd.read_csv(submission_path)
n_predict = submission.values.shape[0]


# submission의 time 항목은 string이므로 이를 timestamp로 변환

submission_time = submission["time"].copy()

for i in range(submission.shape[0]):
    if submission_time[i][-8:] == "24:00:00":
        submission_time[i] = submission_time[i].replace("24:00:00", " 00:00:00")
        submission_time[i] = pd.to_datetime(submission_time[i]) + dt.timedelta(days = 1)

    submission_time[i] = pd.Timestamp(submission_time[i])

submission_time = submission_time.astype({"time":"object"})

# month, day, hour 추가
month = []
day = []
hour = []

for i in range(len(submission_time)):
    month.append(submission_time[i].month)
    day.append(submission_time[i].day)
    hour.append(submission_time[i].hour)

month = np.array(month).reshape(-1,1)
day = np.array(day).reshape(-1,1)
hour = np.array(hour).reshape(-1,1)

d_date = (month, day, hour)

# fcst data for submission
# 2월 데이터 

x_name_fcst = ["temp_fcst", "ws_fcst", "humid_fcst"]
'''
ulsan_fcst_data, ulsan_fcst_scaler = preprocessing(ulsan_fcst_data, x_name_fcst)
dangjin_fcst_data, dangjin_fcst_scaler = preprocessing(dangjin_fcst_data, x_name_fcst)
'''

# submission for ulsan
x_name = ["temp_obs","ws_obs","humid_obs","ulsan"]
y_name = ["ulsan"]
yhats_ulsan = None

for n_in, model in zip(n_in_list, ulsan_models):
    d_obs = ulsan_data_for_predict[x_name][- n_in : ].values.reshape(1,-1)
    d_fcst = ulsan_fcst_data[x_name_fcst][-24*28 - 24 * 3: - 24 *3].values.reshape(24 * 28, len(x_name_fcst))
    prediction = submission_predict2(model, (d_obs, d_fcst), n_predict = 24 * 28, model_type = "mv",  x_date = d_date)
    yhat = prediction.reshape(-1,1)
    
    #yhat = PG_inverse(yhat, ulsan_scaler)

    if yhats_ulsan is None:
        yhats_ulsan = yhat
    else:
        yhats_ulsan = np.concatenate((yhats_ulsan, yhat), axis = 1)

ulsan_weighted_sum = np.dot(yhats_ulsan, w_ulsan).reshape(-1,1)
submission.iloc[0:24*28, 4] = ulsan_weighted_sum

# submission for dangjin_floating
x_name = ["temp_obs","ws_obs","humid_obs","dangjin_floating"]
y_name = ["dangjin_floating"]
yhats_dangjin_floating = None

for n_in, model in zip(n_in_list, dangjin_floating_models):
    d_obs = dangjin_data_for_predict[x_name][- n_in : ].values.reshape(1,-1)
    d_fcst = dangjin_fcst_data[x_name_fcst][-24*28 - 24 * 3: - 24 *3].values.reshape(24 * 28, len(x_name_fcst))
    prediction = submission_predict2(model, (d_obs, d_fcst), n_predict = 24 * 28, model_type = "mv",  x_date = d_date)
    yhat = prediction.reshape(-1,1)
    
    #yhat = PG_inverse(yhat, dangjin_floating_scaler)

    if yhats_dangjin_floating is None:
        yhats_dangjin_floating = yhat
    else:
        yhats_dangjin_floating = np.concatenate((yhats_dangjin_floating, yhat), axis = 1)

dangjin_floating_weighted_sum = np.dot(yhats_dangjin_floating, w_dangjin_floating).reshape(-1,1)
submission.iloc[0:24*28, 1] = dangjin_floating_weighted_sum


# submission for dangjin_warehouse
x_name = ["temp_obs","ws_obs","humid_obs","dangjin_warehouse"]
y_name = ["dangjin_warehouse"]
yhats_dangjin_warehouse = None

for n_in, model in zip(n_in_list, dangjin_warehouse_models):
    d_obs = dangjin_data_for_predict[x_name][- n_in : ].values.reshape(1,-1)
    d_fcst = dangjin_fcst_data[x_name_fcst][-24*28 - 24 * 3: - 24 *3].values.reshape(24 * 28, len(x_name_fcst))
    prediction = submission_predict2(model, (d_obs, d_fcst), n_predict = 24 * 28, model_type = "mv",  x_date = d_date)
    yhat = prediction.reshape(-1,1)
    
    #yhat = PG_inverse(yhat, dangjin_warehouse_scaler)

    if yhats_dangjin_warehouse is None:
        yhats_dangjin_warehouse = yhat
    else:
        yhats_dangjin_warehouse = np.concatenate((yhats_dangjin_warehouse, yhat), axis = 1)

dangjin_warehouse_weighted_sum = np.dot(yhats_dangjin_warehouse, w_dangjin_warehouse).reshape(-1,1)
submission.iloc[0:24*28, 2] = dangjin_warehouse_weighted_sum

# submission for dangjin
x_name = ["temp_obs","ws_obs","humid_obs","dangjin"]
y_name = ["dangjin"]
yhats_dangjin = None

for n_in, model in zip(n_in_list, dangjin_models):
    d_obs = dangjin_data_for_predict[x_name][- n_in : ].values.reshape(1,-1)
    d_fcst = dangjin_fcst_data[x_name_fcst][-24*28 - 24 * 3: - 24 *3].values.reshape(24 * 28, len(x_name_fcst))
    prediction = submission_predict2(model, (d_obs, d_fcst), n_predict = 24 * 28, model_type = "mv",  x_date = d_date)
    yhat = prediction.reshape(-1,1)
    
    #yhat = PG_inverse(yhat, dangjin_scaler)

    if yhats_dangjin is None:
        yhats_dangjin = yhat
    else:
        yhats_dangjin = np.concatenate((yhats_dangjin, yhat), axis = 1)

dangjin_weighted_sum = np.dot(yhats_dangjin, w_dangjin).reshape(-1,1)
submission.iloc[0:24*28, 3] = dangjin_weighted_sum


submission.to_csv("submission.csv", index = False)