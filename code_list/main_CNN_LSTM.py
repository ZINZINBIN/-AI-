import os
import pickle
import numpy as np
import pandas as pd
import scipy as sp
import tensorflow as tf
import matplotlib.pyplot as plt
import glob
from lib_CNN_LSTM import *

# load data from pickle
with open('./witt_preprocessing/pickles/dangjin_merged.pkl','rb') as f:
    dangjin_data = pickle.load(f)
with open('./witt_preprocessing/pickles/ulsan_merged.pkl','rb') as f:
    ulsan_data = pickle.load(f)

# time as index
dangjin_data.set_index('time', inplace=True)
ulsan_data.set_index('time', inplace=True)

# preprocessing
col_name = ["Temperature_obs", "Wind_X_obs", "Wind_Y_obs", "Humidity_obs", "Cloud_obs"]
ulsan_data, ulsan_scaler_mv = preprocessing(ulsan_data, col_name)
dangjin_data, dangjin_scaler_mv = preprocessing(dangjin_data, col_name)

# model weights path
weights_path = "./model_weights/"
ulsan_weights_path = weights_path + "ulsan_weights.h5"
dangjin_floating_weights_path = weights_path + "dangjin_floating_weights.h5"
dangjin_warehouse_weights_path = weights_path + "dangjin_warehouse_weights.h5"
dangjin_weights_path = weights_path + "dangjin_weights.h5"

# load pre-trained model(from colab)
ulsan_model = build_CNN_LSTM(input_shape = (1,24,9), filters = 128, kernel_size = 3, strides = 1, pool_size = 2, dropout = 0.2, units = 256, n_predict = 24)
dangjin_floating_model = build_CNN_LSTM(input_shape = (1,24,9), filters = 128, kernel_size = 3, strides = 1, pool_size = 2, dropout = 0.2, units = 512, n_predict = 24)
dangjin_warehouse_model = build_CNN_LSTM(input_shape = (1,24,9), filters = 128, kernel_size = 3, strides = 1, pool_size = 2, dropout = 0.2, units = 512, n_predict = 24)
dangjin_model = build_CNN_LSTM(input_shape = (1,24,9), filters = 128, kernel_size = 3, strides = 1, pool_size = 2, dropout = 0.2, units = 512, n_predict = 24)

ulsan_model.load_weights(ulsan_weights_path)
dangjin_floating_model.load_weights(dangjin_floating_weights_path)
dangjin_warehouse_model.load_weights(dangjin_warehouse_weights_path)
dangjin_model.load_weights(dangjin_weights_path)

# submission 
submission_path = "./submission.csv"
submission = pd.read_csv(submission_path, encoding = "CP949")

ulsan_obs_feb_path = "./original_dataset/external_data/ulsan_obs_2021-02.csv" 
dangjin_obs_feb_path = "./original_dataset/external_data/dangjin_obs_2021-02.csv"

ulsan_obs_feb = pd.read_csv(ulsan_obs_feb_path, encoding = "CP949" ) 
dangjin_obs_feb = pd.read_csv(dangjin_obs_feb_path, encoding = "CP949")

dangjin_obs_feb.rename(
    columns = {
        "일시":"time",
        "기온(°C)":"Temperature",
        "풍속(m/s)":"WindSpeed",
        "풍향(16방위)":"WindDirection",
        "습도(%)":"Humidity",
        "전운량(10분위)":"Cloud"
    }, inplace = True)

ulsan_obs_feb.rename(
    columns = {
        "일시":"time",
        "기온(°C)":"Temperature",
        "풍속(m/s)":"WindSpeed",
        "풍향(16방위)":"WindDirection",
        "습도(%)":"Humidity",
        "전운량(10분위)":"Cloud"
    }, inplace = True)

dangjin_obs_feb = dangjin_obs_feb.drop(columns = ["지점", "지점명"])
ulsan_obs_feb = ulsan_obs_feb.drop(columns = ["지점","지점명"])

dangjin_obs_feb = dangjin_obs_feb.join(preprocess_wind(dangjin_obs_feb))
ulsan_obs_feb = ulsan_obs_feb.join(preprocess_wind(ulsan_obs_feb))

for i in range(dangjin_obs_feb.shape[0]):
    dangjin_obs_feb["time"][i] = pd.to_datetime(dangjin_obs_feb["time"][i])
    
for i in range(ulsan_obs_feb.shape[0]):
    ulsan_obs_feb["time"][i] = pd.to_datetime(ulsan_obs_feb["time"][i])
    
dangjin_obs_feb = dangjin_obs_feb.astype({"time":"object"})
ulsan_obs_feb = ulsan_obs_feb.astype({"time":"object"})

# add seasonality
dangjin_obs_feb = add_seasonality(dangjin_obs_feb)
ulsan_obs_feb = add_seasonality(ulsan_obs_feb)

# data preprocessing
col_name = ["Temperature", "Wind_X", "Wind_Y", "Humidity", "Cloud"]
tmp_dangjin = dangjin_scaler_mv.transform(dangjin_obs_feb[col_name].values)
tmp_ulsan = ulsan_scaler_mv.transform(dangjin_obs_feb[col_name].values)

dangjin_obs_feb = dangjin_obs_feb.drop(columns = col_name)
dangjin_obs_feb[col_name] = tmp_dangjin

ulsan_obs_feb = ulsan_obs_feb.drop(columns = col_name)
ulsan_obs_feb[col_name] = tmp_ulsan

# ulsan forecasting
x_name_fcst = ["Day_cos","Day_sin","Year_cos","Year_sin","Temperature", "Wind_X", "Wind_Y", "Humidity", "Cloud"]
last_row = ulsan_obs_feb.iloc[-1,:]
ulsan_obs_feb.loc[len(ulsan_obs_feb)] = last_row

input_fcst = ulsan_obs_feb[x_name_fcst].iloc[1:,:].values.reshape(27, 1, 24, 9)
prediction = ulsan_model.predict(input_fcst).reshape(-1,1)
submission.iloc[0:24*27,4] = prediction

# dangjin_floating forecasting
x_name_fcst = ["Day_cos","Day_sin","Year_cos","Year_sin","Temperature", "Wind_X", "Wind_Y", "Humidity", "Cloud"]
last_row = dangjin_obs_feb.iloc[-1,:]
dangjin_obs_feb.loc[len(dangjin_obs_feb)] = last_row

input_fcst = dangjin_obs_feb[x_name_fcst].iloc[1:,:].values.reshape(27, 1, 24, 9)
prediction = dangjin_floating_model.predict(input_fcst).reshape(-1,1)
submission.iloc[0:24*27,1] = prediction

# dangjin_warehouse forecasting
input_fcst = dangjin_obs_feb[x_name_fcst].iloc[1:,:].values.reshape(27, 1, 24, 9)
prediction = dangjin_warehouse_model.predict(input_fcst).reshape(-1,1)
submission.iloc[0:24*27,2] = prediction

# dangjin_warehouse forecasting
input_fcst = dangjin_obs_feb[x_name_fcst].iloc[1:,:].values.reshape(27, 1, 24, 9)
prediction = dangjin_model.predict(input_fcst).reshape(-1,1)
submission.iloc[0:24*27,3] = prediction

submission.to_csv("submission_CNN_LSTM.csv", index = False)