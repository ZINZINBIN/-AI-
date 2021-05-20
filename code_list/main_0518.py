'''
# ======================================================================== #
# =========================== File Explanation =========================== #
# ======================================================================== #

- model: xgboost
- preprocessing: dropna, (wd, ws) -> (ws_x, ws_y), month, day, hour -> (m,d,h)
- 구성: main, lib_function
'''

# library
import os
import numpy as np
import pandas as pd
import scipy as sp
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import glob
from lib_function_0518 import *
import pickle

# ======================================================================== #
# ===================== load data and preprocessing ====================== #
# ======================================================================== #

# load data
with open('./witt_preprocessing/pickles/dangjin_merged.pkl','rb') as f:
    dangjin_data = pickle.load(f)
with open('./witt_preprocessing/pickles/ulsan_merged.pkl','rb') as f:
    ulsan_data = pickle.load(f)

# data의 month,day, time을 추가

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

dangjin_data["month"] = dangjin_data["month"].astype(int)
dangjin_data["day"] = dangjin_data["day"].astype(int)
dangjin_data["hour"] = dangjin_data["hour"].astype(int)

ulsan_data["month"] = ulsan_data["month"].astype(int)
ulsan_data["day"] = ulsan_data["day"].astype(int)
ulsan_data["hour"] = ulsan_data["hour"].astype(int)

# time as index
dangjin_data.set_index('time', inplace=True)
ulsan_data.set_index('time', inplace=True)

# ======================================================================== #
# ================== build model and training(xgboost) =================== #
# ======================================================================== #

# model architecture
import xgboost as xgb
esr = 400 # early stopping round

params = {
    "booster":"dart",
    #objective":"reg:pseudohubererror",
    "objective":"reg:squarederror",
    "learning_rate":0.3,
    #"max_depth":7,
    "max_depth":9,
    "n_estimators":1024,
    "nthread":-1,
    "gamma":10.0,
    "subsample":1.0,
    "colsample_bytree":1.0,
    "colsample_bylevel":1.0,
    "min_child_weight":5, #5
    "reg_lambda":0.1,
    "reg_alpha":0.1, # 1.0
    "sample_type":"weighted"
}

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


def data_generate_xgb(data, x_name, y_name, n_out = 1, dropnan = False, test_size = 0.3):
    data_copy = data.copy()[x_name]
    col_fcst = list()
    name_fcst = list()
    
    for i in range(1, n_out + 1):
        y = data.copy()[y_name]
        col_fcst.append(y.shift(-i))

        if i == 1:
            name_fcst += [("%s(t+1)"%(name)) for name in y_name]
        else:
            name_fcst += [("%s(t+%d)"%(name, i)) for name in y_name]
        
    agg = pd.concat(col_fcst, axis = 1)
    agg.columns = name_fcst

    result = pd.concat([data_copy, agg], axis = 1)

    if dropnan:
        result.dropna(inplace = True)

    del agg
    del data_copy

    x = result[x_name].values.reshape(-1, len(x_name))
    y = result[name_fcst].values.reshape(-1,1)

    total_len = len(result)
    test_len = int(total_len * test_size)
    
    x_train, x_val, y_train, y_val = x[:-test_len,:],x[-test_len:,:], y[:-test_len], y[-test_len:]
    
    return x_train, x_val, y_train, y_val

# ulsan 
x_name = ["Day_cos","Day_sin","Year_cos","Year_sin","Temperature_obs", "Humidity_obs", "Wind_X_obs", "Wind_Y_obs", "Cloud_obs", "ulsan"]
y_name = ["ulsan"]
ulsan_model = build_xgb(params)
x_train, x_val, y_train, y_val = data_generate_xgb(ulsan_data, x_name, y_name, dropnan = True, test_size = 0.3)
custom_eval_ulsan = lambda x,y : custom_evaluation(x, y, cap = "ulsan")

ulsan_model.fit(x_train, y_train, eval_set = [(x_val, y_val)], early_stopping_rounds = esr, eval_metric = custom_eval_ulsan)

# dangjin floating
x_name = ["Day_cos","Day_sin","Year_cos","Year_sin","Temperature_obs", "Humidity_obs", "Wind_X_obs", "Wind_Y_obs","Cloud_obs", "dangjin_floating"]
y_name = ["dangjin_floating"]
dangjin_floating_model = build_xgb(params)
x_train, x_val, y_train, y_val = data_generate_xgb(dangjin_data, x_name, y_name, dropnan = True, test_size = 0.3)
custom_eval_dangjin_floating = lambda x,y : custom_evaluation(x, y, cap = "dangjin_floating")
dangjin_floating_model.fit(x_train, y_train, eval_set = [(x_val, y_val)], early_stopping_rounds = esr, eval_metric = custom_eval_dangjin_floating)

# dangjin warehouse
x_name = ["Day_cos","Day_sin","Year_cos","Year_sin","Temperature_obs", "Humidity_obs", "Wind_X_obs", "Wind_Y_obs", "Cloud_obs","dangjin_warehouse"]
y_name = ["dangjin_warehouse"]
dangjin_warehouse_model = build_xgb(params)
x_train, x_val, y_train, y_val = data_generate_xgb(dangjin_data, x_name, y_name, dropnan = True, test_size = 0.3)
custom_eval_dangjin_warehouse = lambda x,y : custom_evaluation(x, y, cap = "dangjin_warehouse")
dangjin_warehouse_model.fit(x_train, y_train, eval_set = [(x_val, y_val)], early_stopping_rounds = esr, eval_metric = custom_eval_dangjin_warehouse)

# dangjin 
x_name = ["Day_cos","Day_sin","Year_cos","Year_sin","Temperature_obs", "Humidity_obs", "Wind_X_obs", "Wind_Y_obs","Cloud_obs", "dangjin"]
y_name = ["dangjin"]
dangjin_model = build_xgb(params)
x_train, x_val, y_train, y_val = data_generate_xgb(dangjin_data, x_name, y_name, dropnan = True, test_size = 0.3)
custom_eval_dangjin = lambda x,y : custom_evaluation(x, y, cap = "dangjin")
dangjin_model.fit(x_train, y_train, eval_set = [(x_val, y_val)], early_stopping_rounds = esr, eval_metric = custom_eval_dangjin)


# ======================================================================== #
# ================= forecasting and evaluate the model =================== #
# ======================================================================== #

# evaluation
term_3d = range(0, 24 * 3)
term_7d = range(0, 24 * 7)
term_30d = range(0, 24 * 30)

# ulsan evaluation
x_name_fcst = ["Day_cos","Day_sin","Year_cos","Year_sin","Temperature_obs", "Humidity_obs", "Wind_X_obs", "Wind_Y_obs", "Cloud_obs",]
x_name = ["Day_cos","Day_sin","Year_cos","Year_sin","Temperature_obs", "Humidity_obs", "Wind_X_obs", "Wind_Y_obs", "Cloud_obs","ulsan"]
y_name = ["ulsan"]

n_in = 1
n_predict = 24 * 30
start_data_in = ulsan_data[x_name].iloc[-24*30*1 - n_in].values.reshape(1,-1)
fcst_data = ulsan_data[x_name_fcst].iloc[-24 * 30 * 1 : ].values.reshape(-1, len(x_name_fcst))

# comparing result
def submission_predict_xgb(model, n_predict, start_data_in, fcst_data):
    y_preds = []
    for i in range(0, n_predict):
        if i == 0:
            y_pred = model.predict(start_data_in)
            data_in = np.append(fcst_data[i,:].reshape(1,-1), y_pred).reshape(1,-1)
            y_preds.append(y_pred)
        else:
            y_pred = model.predict(data_in)
            data_in = np.append(fcst_data[i,:].reshape(1,-1), y_pred).reshape(1,-1)
            y_preds.append(y_pred)
    total_prediction = np.array(y_preds).reshape(-1,1)

    return total_prediction

prediction = submission_predict_xgb(ulsan_model, n_predict = n_predict, start_data_in = start_data_in, fcst_data = fcst_data)
yhat = prediction.reshape(-1,1)
yreal = ulsan_data[y_name].iloc[-24*30*1 : ].values.reshape(-1,1)


for i, term in enumerate([term_3d, term_7d, term_30d]):
    name = str(len(term) / 24) + " - days forecast: ulsan"
    plt.figure(i+1, figsize = (10,5))
    plt.plot(yreal[term], label = "real")
    plt.plot(yhat[term], label = "forecast")
    plt.ylabel("ulsan, unit:None")
    plt.title(name)
    plt.legend()
    plt.show()


# dangjin_floating evaluation
x_name_fcst = ["Day_cos","Day_sin","Year_cos","Year_sin","Temperature_obs", "Humidity_obs", "Wind_X_obs", "Wind_Y_obs","Cloud_obs"]
x_name = ["Day_cos","Day_sin","Year_cos","Year_sin","Temperature_obs", "Humidity_obs", "Wind_X_obs", "Wind_Y_obs","Cloud_obs", "dangjin_floating"]
y_name = ["dangjin_floating"]

n_in = 1
n_predict = 24 * 30
start_data_in = dangjin_data[x_name].iloc[-24*30*1 - n_in].values.reshape(1,-1)
fcst_data = dangjin_data[x_name_fcst].iloc[-24 * 30 * 1 : ].values.reshape(-1, len(x_name_fcst))

prediction = submission_predict_xgb(dangjin_floating_model, n_predict = n_predict, start_data_in = start_data_in, fcst_data = fcst_data)
yhat = prediction.reshape(-1,1)
yreal = dangjin_data[y_name].iloc[-24*30*1 : ].values.reshape(-1,1)

for i, term in enumerate([term_3d, term_7d, term_30d]):
    name = str(len(term) / 24) + " - days forecast: dangjin_floating"
    plt.figure(i+1, figsize = (10,5))
    plt.plot(yreal[term], label = "real")
    plt.plot(yhat[term], label = "forecast")
    plt.ylabel("dangjin_floating, unit:None")
    plt.title(name)
    plt.legend()
    plt.show()

# dangjin_warehouse evaluation
x_name_fcst = ["Day_cos","Day_sin","Year_cos","Year_sin","Temperature_obs", "Humidity_obs", "Wind_X_obs", "Wind_Y_obs" , "Cloud_obs"]
x_name = ["Day_cos","Day_sin","Year_cos","Year_sin","Temperature_obs", "Humidity_obs", "Wind_X_obs", "Wind_Y_obs", "Cloud_obs","dangjin_warehouse"]
y_name = ["dangjin_warehouse"]

n_in = 1
n_predict = 24 * 30
start_data_in = dangjin_data[x_name].iloc[-24*30*1 - n_in].values.reshape(1,-1)
fcst_data = dangjin_data[x_name_fcst].iloc[-24 * 30 * 1 : ].values.reshape(-1, len(x_name_fcst))

prediction = submission_predict_xgb(dangjin_warehouse_model, n_predict = n_predict, start_data_in = start_data_in, fcst_data = fcst_data)
yhat = prediction.reshape(-1,1)
yreal = dangjin_data[y_name].iloc[-24*30*1 : ].values.reshape(-1,1)

for i, term in enumerate([term_3d, term_7d, term_30d]):
    name = str(len(term) / 24) + " - days forecast: dangjin_warehouse"
    plt.figure(i+1, figsize = (10,5))
    plt.plot(yreal[term], label = "real")
    plt.plot(yhat[term], label = "forecast")
    plt.ylabel("dangjin_warehouse, unit:None")
    plt.title(name)
    plt.legend()
    plt.show()

# dangjin evaluation
x_name_fcst = ["Day_cos","Day_sin","Year_cos","Year_sin","Temperature_obs", "Humidity_obs", "Wind_X_obs", "Wind_Y_obs", "Cloud_obs"]
x_name = ["Day_cos","Day_sin","Year_cos","Year_sin","Temperature_obs", "Humidity_obs", "Wind_X_obs", "Wind_Y_obs", "Cloud_obs","dangjin"]
y_name = ["dangjin"]

n_in = 1
n_predict = 24 * 30
start_data_in = dangjin_data[x_name].iloc[-24*30*1 - n_in].values.reshape(1,-1)
fcst_data = dangjin_data[x_name_fcst].iloc[-24 * 30 * 1 : ].values.reshape(-1, len(x_name_fcst))

prediction = submission_predict_xgb(dangjin_model, n_predict = n_predict, start_data_in = start_data_in, fcst_data = fcst_data)
yhat = prediction.reshape(-1,1)
yreal = dangjin_data[y_name].iloc[-24*30*1 : ].values.reshape(-1,1)

for i, term in enumerate([term_3d, term_7d, term_30d]):
    name = str(len(term) / 24) + " - days forecast: dangjin"
    plt.figure(i+1, figsize = (10,5))
    plt.plot(yreal[term], label = "real")
    plt.plot(yhat[term], label = "forecast")
    plt.ylabel("dangjin, unit:None")
    plt.title(name)
    plt.legend()
    plt.show()

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

def preprocess_wind(data):
    '''
    data: pd.DataFrmae which contains the columns 'WindSpeed' and 'WindDirection'
    '''

    # degree to radian
    wind_direction_radian = data['WindDirection'] * np.pi / 180

    # polar coordinate to cartesian coordinate
    wind_x = data['WindSpeed'] * np.cos(wind_direction_radian)
    wind_y = data['WindDirection'] * np.sin(wind_direction_radian)

    # name pd.series
    wind_x.name = 'Wind_X'
    wind_y.name = 'Wind_Y'

    return wind_x, wind_y

dangjin_obs_feb = dangjin_obs_feb.join(preprocess_wind(dangjin_obs_feb))
ulsan_obs_feb = ulsan_obs_feb.join(preprocess_wind(ulsan_obs_feb))


for i in range(dangjin_obs_feb.shape[0]):
    dangjin_obs_feb["time"][i] = pd.to_datetime(dangjin_obs_feb["time"][i])
    
for i in range(ulsan_obs_feb.shape[0]):
    ulsan_obs_feb["time"][i] = pd.to_datetime(ulsan_obs_feb["time"][i])
    
dangjin_obs_feb = dangjin_obs_feb.astype({"time":"object"})
ulsan_obs_feb = ulsan_obs_feb.astype({"time":"object"})

# add seasonality
def day_of_year(datetime): #pd.datetime
    return pd.Period(datetime, freq='D').dayofyear # day_of_year와 같은데 이상하게 작동이 안됨.

def add_seasonality(df):
    new_df = df.copy()
    
    new_df['Day_cos'] = new_df['time'].apply(lambda x: np.cos(x.hour * (2 * np.pi) / 24))
    new_df['Day_sin'] = new_df['time'].apply(lambda x: np.sin(x.hour * (2 * np.pi) / 24))

    new_df['Year_cos'] = new_df['time'].apply(lambda x: np.cos(day_of_year(x) * (2 * np.pi) / 365))
    new_df['Year_sin'] = new_df['time'].apply(lambda x: np.sin(day_of_year(x) * (2 * np.pi) / 365))

    return new_df

dangjin_obs_feb = add_seasonality(dangjin_obs_feb)
ulsan_obs_feb = add_seasonality(ulsan_obs_feb)




    
'''
# month, day, hour addition for obs_feb
month = []
day = []
hour = []

for i in range(len(dangjin_obs_feb)):
    month.append(dangjin_obs_feb["time"][i].month)
    day.append(dangjin_obs_feb["time"][i].day)
    hour.append(dangjin_obs_feb["time"][i].hour)

month = np.array(month).reshape(-1,1)
day = np.array(day).reshape(-1,1)
hour = np.array(hour).reshape(-1,1)

dangjin_obs_feb["month"] = month
dangjin_obs_feb["day"] = day
dangjin_obs_feb["hour"] = hour

dangjin_obs_feb["month"] = dangjin_obs_feb["month"].astype(int)
dangjin_obs_feb["day"] = dangjin_obs_feb["day"].astype(int)
dangjin_obs_feb["hour"] = dangjin_obs_feb["hour"].astype(int)


month = []
day = []
hour = []

for i in range(len(ulsan_obs_feb)):
    month.append(ulsan_obs_feb["time"][i].month)
    day.append(ulsan_obs_feb["time"][i].day)
    hour.append(ulsan_obs_feb["time"][i].hour)

month = np.array(month).reshape(-1,1)
day = np.array(day).reshape(-1,1)
hour = np.array(hour).reshape(-1,1)

ulsan_obs_feb["month"] = month
ulsan_obs_feb["day"] = day
ulsan_obs_feb["hour"] = hour

ulsan_obs_feb["month"] = ulsan_obs_feb["month"].astype(int)
ulsan_obs_feb["day"] = ulsan_obs_feb["day"].astype(int)
ulsan_obs_feb["hour"] = ulsan_obs_feb["hour"].astype(int)
'''

# ulsan forecasting
x_name_fcst = ["Day_cos","Day_sin","Year_cos","Year_sin","Temperature", "Humidity", "Wind_X", "Wind_Y", "Cloud"]

start_data_in = np.append(ulsan_obs_feb[x_name_fcst].iloc[0].values.reshape(1,-1),0).reshape(1,-1)
obs_data = ulsan_obs_feb[x_name_fcst].iloc[1:,:].values.reshape(-1, len(x_name_fcst))
prediction = submission_predict_xgb(ulsan_model, n_predict = 24 * 27 - 1, start_data_in = start_data_in, fcst_data = obs_data)
yhat = prediction.reshape(-1,1)
submission.iloc[0:24*27 -1,4] = yhat

# dangjin_floating forecasting
x_name_fcst = ["Day_cos","Day_sin","Year_cos","Year_sin","Temperature", "Humidity", "Wind_X", "Wind_Y", "Cloud"]

start_data_in = np.append(dangjin_obs_feb[x_name_fcst].iloc[0].values.reshape(1,-1),0).reshape(1,-1)
obs_data = dangjin_obs_feb[x_name_fcst].iloc[1:,:].values.reshape(-1, len(x_name_fcst))
prediction = submission_predict_xgb(dangjin_floating_model, n_predict = 24 * 27 - 1, start_data_in = start_data_in, fcst_data = obs_data)
yhat = prediction.reshape(-1,1)
submission.iloc[0:24*27 -1,1] = yhat

# dangjin_warehouse forecasting
x_name_fcst = ["Day_cos","Day_sin","Year_cos","Year_sin","Temperature", "Humidity", "Wind_X", "Wind_Y", "Cloud"]

start_data_in = np.append(dangjin_obs_feb[x_name_fcst].iloc[0].values.reshape(1,-1),0).reshape(1,-1)
obs_data = dangjin_obs_feb[x_name_fcst].iloc[1:,:].values.reshape(-1, len(x_name_fcst))
prediction = submission_predict_xgb(dangjin_warehouse_model, n_predict = 24 * 27 - 1, start_data_in = start_data_in, fcst_data = obs_data)
yhat = prediction.reshape(-1,1)
submission.iloc[0:24*27 -1,2] = yhat

# dangjin forecasting
x_name_fcst = ["Day_cos","Day_sin","Year_cos","Year_sin","Temperature", "Humidity", "Wind_X", "Wind_Y", "Cloud"]

start_data_in = np.append(dangjin_obs_feb[x_name_fcst].iloc[0].values.reshape(1,-1),0).reshape(1,-1)
obs_data = dangjin_obs_feb[x_name_fcst].iloc[1:,:].values.reshape(-1, len(x_name_fcst))
prediction = submission_predict_xgb(dangjin_model, n_predict = 24 * 27 - 1, start_data_in = start_data_in, fcst_data = obs_data)
yhat = prediction.reshape(-1,1)
submission.iloc[0:24*27 -1,3] = yhat

submission.to_csv("submission.csv", index = False)