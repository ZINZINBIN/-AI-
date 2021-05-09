'''
# ================================================================================ #
# =========================== Goolge Colab File Upload =========================== #
# ================================================================================ #

from google.colab import drive
drive.mount('/content/drive')

from google.colab import output
# !cp 파일1 파일2 # 파일1을 파일2로 복사 붙여넣기
!cp "/content/drive/MyDrive/Colab Notebooks/동서발전 태양광 발전량 예측 경진대회/data.zip" "data.zip"
# data.zip을 현재 디렉터리에 압축해제
!unzip "data.zip"
output.clear()

import os


# path
path = "./"
dangjin_fcst_data_path = path + "dangjin_fcst_data.csv"
dangjin_obs_data_path = path + "dangjin_obs_data.csv"
energy_data_path = path + "energy.csv"
ulsan_fcst_data_path = path + "ulsan_fcst_data.csv"
ulsan_obs_data_path = path + "ulsan_obs_data.csv"
site_info_path = path + "site_info.csv"

# import library 
import numpy as np
import pandas as pd
import scipy as sp
import tensorflow as tf
#import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import os
import glob

# file convert to pandas data
dangjin_fcst_data = pd.read_csv(dangjin_fcst_data_path)
dangjin_obs_data = pd.read_csv(dangjin_obs_data_path)
energy_data = pd.read_csv(energy_data_path)
ulsan_fcst_data = pd.read_csv(ulsan_fcst_data_path)
ulsan_obs_data = pd.read_csv(ulsan_obs_data_path)
site_info = pd.read_csv(site_info_path)


'''
# ======================================================================== #
# ========================== From gitlab Clone =========================== #
# ======================================================================== #

import os
import numpy as np
import pandas as pd
import scipy as sp
import tensorflow as tf
import matplotlib.pyplot as plt
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
dangjin_fcst_data = dangjin_fcst_data.groupby("time_fcst", as_index = False).mean()
dangjin_fcst_data = dangjin_fcst_data.drop(columns = ["forecast_fcst"])

ulsan_fcst_data["time_fcst"] = pd.to_datetime(ulsan_fcst_data["time"].copy()) + ulsan_fcst_data["forecast_fcst"].copy().astype("timedelta64[h]")
ulsan_fcst_data = ulsan_fcst_data.groupby("time_fcst", as_index = False).mean()
ulsan_fcst_data = ulsan_fcst_data.drop(columns = ["forecast_fcst"])

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

# total dataset summary
display(dangjin_data)
display(ulsan_data)


# model architecture

import xgboost as xgb

params = {
    "booster":"dart",
    "objective":"reg:pseudohubererror",
    #"objective":"reg:squarederror",
    "learning_rate":0.3,
    #"max_depth":7,
    "max_depth":9,
    "n_estimators":1000,
    "nthread":-1,
    "gamma":1.0,
    "subsample":0.7,
    "colsample_bytree":1.0,
    "colsample_bylevel":1.0,
    "min_child_weight":5,
    "reg_lambda":0.1,
    "reg_alpha":1.0
}

n_out = 1
ratio = 0.8 # train size ratio
n_in_list = [24 * 15, 24 * 30, 24 * 30 * 3, 24 * 30 * 12]
esr = 200 # early stopping round

# ulsan model
x_name = ["ulsan"]
y_name = ["ulsan"]
n_features = len(x_name)

ulsan_models = [build_xgb(params) for n_in in n_in_list]
for model, n_in in zip(ulsan_models, n_in_list):
    (x_train, y_train), (x_test, y_test) = data_generator(ulsan_data, n_in, n_out, ratio, x_name, y_name)
    model.fit(x_train, y_train, eval_set = [(x_test, y_test)], early_stopping_rounds = esr)
    
    del x_train, x_test, y_train, y_test

# dangjin_floating_models training

x_name = ["dangjin_floating"]
y_name = ["dangjin_floating"]
dangjin_floating_models = [build_xgb(params) for n_in in n_in_list]

for model, n_in in zip(dangjin_floating_models, n_in_list):
    (x_train, y_train), (x_test, y_test) = data_generator(dangjin_data, n_in, n_out, ratio, x_name, y_name)
    model.fit(x_train, y_train, eval_set = [(x_test, y_test)], early_stopping_rounds = esr)
    del x_train, x_test, y_train, y_test


# dangjin_warehouse_models training

x_name = ["dangjin_warehouse"]
y_name = ["dangjin_warehouse"]
dangjin_warehouse_models = [build_xgb(params) for n_in in n_in_list]

for model, n_in in zip(dangjin_warehouse_models, n_in_list):
    (x_train, y_train), (x_test, y_test) = data_generator(dangjin_data, n_in, n_out, ratio, x_name, y_name)
    model.fit(x_train, y_train, eval_set = [(x_test, y_test)], early_stopping_rounds = esr)
    del x_train, x_test, y_train, y_test


# dangjin_models training

x_name = ["dangjin"]
y_name = ["dangjin"]
dangjin_models = [build_xgb(params) for n_in in n_in_list]

for model, n_in in zip(dangjin_models, n_in_list):
    (x_train, y_train), (x_test, y_test) = data_generator(dangjin_data, n_in, n_out, ratio, x_name, y_name)
    model.fit(x_train, y_train, eval_set = [(x_test, y_test)], early_stopping_rounds = esr)
    del x_train, x_test, y_train, y_test

# forecasting
# 21.01.01 - 21.01.31
# ulsan, dangjin_floating, dangjin_warehouse, dangjin Power Generation prediction

term_3d = range(0, 24 * 3)
term_7d = range(0, 24 * 7)
term_30d = range(0, 24 * 30)

# ulsan prediction
x_name = ["ulsan"]
y_name = ["ulsan"]
yhats_ulsan = None

for n_in, model in zip(n_in_list, ulsan_models):
    name = "ulsan, timesteps: " + str(n_in / 24) + "-day"
    d_obs = ulsan_data[x_name][-24*30*1 - n_in : - 24*30*1].values.reshape(1,-1)
    prediction = submission_predict(model, d_obs, model_type = "uv", n_predict = 24 * 30)
    actual = ulsan_data[y_name][-24*30*1:].values
    yreal = actual.reshape(-1,1)
    yhat = prediction.reshape(-1,1)

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
x_name = ["dangjin_floating"]
y_name = ["dangjin_floating"]
yhats_dangjin_floating = None

for n_in, model in zip(n_in_list, dangjin_floating_models):
    name = "dangjin_floating, timesteps: " + str(n_in / 24) + "-day"
    d_obs = dangjin_data[x_name][-24*30*1 - n_in : - 24*30*1].values.reshape(1,-1)
    prediction = submission_predict(model, d_obs, model_type = "uv", n_predict = 24 * 30)
    actual = dangjin_data[y_name][-24*30*1:].values
    yreal = actual.reshape(-1,1)
    yhat = prediction.reshape(-1,1)

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
x_name = ["dangjin_warehouse"]
y_name = ["dangjin_warehouse"]
yhats_dangjin_warehouse = None

for n_in, model in zip(n_in_list, dangjin_warehouse_models):
    name = "dangjin_warehouse, timesteps: " + str(n_in / 24) + "-day"
    d_obs = dangjin_data[x_name][-24*30*1 - n_in : - 24*30*1].values.reshape(1,-1)
    prediction = submission_predict(model, d_obs, model_type = "uv", n_predict = 24 * 30)
    actual = dangjin_data[y_name][-24*30*1:].values
    yreal = actual.reshape(-1,1)
    yhat = prediction.reshape(-1,1)

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
x_name = ["dangjin"]
y_name = ["dangjin"]
yhats_dangjin = None

for n_in, model in zip(n_in_list, dangjin_models):
    name = "dangjin, timesteps: " + str(n_in / 24) + "-day"
    d_obs = dangjin_data[x_name][-24*30*1 - n_in : - 24*30*1].values.reshape(1,-1)
    prediction = submission_predict(model, d_obs, model_type = "uv", n_predict = 24 * 30)
    actual = dangjin_data[y_name][-24*30*1:].values
    yreal = actual.reshape(-1,1)
    yhat = prediction.reshape(-1,1)

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
w_ulsan = ensemble_weights(yhats_ulsan, yreal, model_num)
ulsan_weighted_sum = np.dot(yhats_ulsan, w_ulsan).reshape(-1,1)

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
w_dangjin_floating = ensemble_weights(yhats_dangjin_floating, yreal, model_num)
dangjin_floating_weighted_sum = np.dot(yhats_dangjin_floating, w_dangjin_floating).reshape(-1,1)

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
w_dangjin_warehouse = ensemble_weights(yhats_dangjin_warehouse, yreal, model_num)
dangjin_warehouse_weighted_sum = np.dot(yhats_dangjin_warehouse, w_dangjin_warehouse).reshape(-1,1)

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
w_dangjin = ensemble_weights(yhats_dangjin, yreal, model_num)
dangjin_weighted_sum = np.dot(yhats_dangjin, w_dangjin).reshape(-1,1)

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

# submission for ulsan
x_name = ["ulsan"]
y_name = ["ulsan"]
yhats_ulsan = None

for n_in, model in zip(n_in_list, ulsan_models):
    d_obs = ulsan_data[x_name][- n_in : ].values.reshape(1,-1)
    prediction = submission_predict(model, d_obs, model_type = "uv", n_predict = n_predict)
    yhat = prediction.reshape(-1,1)

    if yhats_ulsan is None:
        yhats_ulsan = yhat
    else:
        yhats_ulsan = np.concatenate((yhats_ulsan, yhat), axis = 1)

ulsan_weighted_sum = np.dot(yhats_ulsan, w_ulsan).reshape(-1,1)
submission["ulsan"] = ulsan_weighted_sum

# submission for dangjin_floating
x_name = ["dangjin_floating"]
y_name = ["dangjin_floating"]
yhats_dangjin_floating = None

for n_in, model in zip(n_in_list, dangjin_floating_models):
    d_obs = dangjin_data[x_name][- n_in : ].values.reshape(1,-1)
    prediction = submission_predict(model, d_obs, model_type = "uv", n_predict = n_predict)
    yhat = prediction.reshape(-1,1)

    if yhats_dangjin_floating is None:
        yhats_dangjin_floating = yhat
    else:
        yhats_dangjin_floating = np.concatenate((yhats_dangjin_floating, yhat), axis = 1)

dangjin_floating_weighted_sum = np.dot(yhats_dangjin_floating, w_dangjin_floating).reshape(-1,1)
submission["dangjin_floating"] = dangjin_floating_weighted_sum

# submission for dangjin_warehouse
x_name = ["dangjin_warehouse"]
y_name = ["dangjin_warehouse"]
yhats_dangjin_warehouse = None

for n_in, model in zip(n_in_list, dangjin_warehouse_models):
    d_obs = dangjin_data[x_name][- n_in : ].values.reshape(1,-1)
    prediction = submission_predict(model, d_obs, model_type = "uv", n_predict = n_predict)
    yhat = prediction.reshape(-1,1)

    if yhats_dangjin_warehouse is None:
        yhats_dangjin_warehouse = yhat
    else:
        yhats_dangjin_warehouse = np.concatenate((yhats_dangjin_warehouse, yhat), axis = 1)

dangjin_warehouse_weighted_sum = np.dot(yhats_dangjin_warehouse, w_dangjin_warehouse).reshape(-1,1)
submission["dangjin_warehouse"] = dangjin_warehouse_weighted_sum

# submission for dangjin
x_name = ["dangjin"]
y_name = ["dangjin"]
yhats_dangjin = None

for n_in, model in zip(n_in_list, dangjin_models):
    d_obs = dangjin_data[x_name][- n_in : ].values.reshape(1,-1)
    prediction = submission_predict(model, d_obs, model_type = "uv", n_predict = n_predict)
    yhat = prediction.reshape(-1,1)

    if yhats_dangjin is None:
        yhats_dangjin = yhat
    else:
        yhats_dangjin = np.concatenate((yhats_dangjin, yhat), axis = 1)

dangjin_weighted_sum = np.dot(yhats_dangjin, w_dangjin).reshape(-1,1)
submission["dangjin"] = dangjin_weighted_sum

submission.to_csv("submission.csv", index = False)