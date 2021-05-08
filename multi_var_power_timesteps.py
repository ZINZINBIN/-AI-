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
import os
import glob

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

# supervised learning을 위한 preprocessing

def series_to_supervised(data, x_name, y_name, n_in, n_out, dropnan = False):

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


# model architecture
#!sudo pip install xgboost
#!sudo pip install --upgrade xgboost

import xgboost as xgb

def data_generator(data, n_in, n_out, ratio, x_name, y_name):
    data_supervised = series_to_supervised(data, x_name, y_name, n_in, n_out, dropnan = True)
    
    x_data = data_supervised.values[:, :-n_out * len(y_name)]
    y_data = data_supervised.values[:, -n_out * len(y_name):]

    data_size = x_data.shape[0]
    train_size = int(data_size * ratio)
    
    x_train = x_data[0:train_size]
    x_test = x_data[train_size:]

    y_train = y_data[0:train_size]
    y_test = y_data[train_size:]

    return (x_train, y_train), (x_test, y_test)

# build model
# build model and parameter setting

def build_xgb(params = None):

    if params is None:
        model = xgb.XGBRegressor()
    else:
        model = xgb.XGBRegressor(
            booster = params["booster"],
            objective = params["objective"],
            learning_rate = params["learning_rate"],    
            max_depth = params["max_depth"],
            n_estimators = params["n_estimators"],
            nthread = params["nthread"],
            gamma = params["gamma"],
            min_child_weight = params["min_child_weight"],
            subsample = params["subsample"],
            reg_lambda = params["reg_lambda"],
            reg_alpha = params["reg_alpha"],
            colsample_bytree = params["colsample_bytree"],
            colsample_bylevel = params["colsample_bylevel"]
        )

    return model

x_name = ["temp_obs","ws_obs","humid_obs","ulsan"]
y_name = ["ulsan"]

n_features = len(x_name)
n_out = 1
ratio = 0.8 # train size ratio
#n_in_list = [24 * 30, 24 * 30 * 3, 24 * 30 * 6, 24 * 30 * 12]

n_in_list = [24 * 3, 24 * 7, 24 * 15, 24 * 30]

# params는 variable마다 각기 다른 optimized tunned value 사용 예정
# grid search는 시간이 걸리므로 일단 단순화된 for문을 이용하여 비교 예정

params = {
    "booster":"dart",
    "objective":"reg:pseudohubererror",
    #"objective":"reg:squarederror",
    "learning_rate":0.1,
    "max_depth":7,
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

esr = 200

# ulsan_models training
ulsan_models = [build_xgb(params) for n_in in n_in_list]
for model, n_in in zip(ulsan_models, n_in_list):
    (x_train, y_train), (x_test, y_test) = data_generator(ulsan_data, n_in, n_out, ratio, x_name, y_name)
    model.fit(x_train, y_train, eval_set = [(x_test, y_test)], early_stopping_rounds = esr)
    
    del x_train, x_test, y_train, y_test


'''
# dangjin_floating_models training
x_name = ["temp_obs","ws_obs","humid_obs","dangjin_floating"]
y_name = ["dangjin_floating"]
dangjin_floating_models = [build_xgb(params) for n_in in n_in_list]

for model, n_in in zip(dangjin_floating_models, n_in_list):
    (x_train, y_train), (x_test, y_test) = data_generator(dangjin_data, n_in, n_out, ratio, x_name, y_name)
    model.fit(x_train, y_train, eval_set = [(x_test, y_test)], early_stopping_rounds = esr)
    del x_train, x_test, y_train, y_test

# dangjin_warehouse_models training
x_name = ["temp_obs","ws_obs","humid_obs","dangjin_warehouse"]
y_name = ["dangjin_warehouse"]
dangjin_floating_models = [build_xgb(params) for n_in in n_in_list]

for model, n_in in zip(dangjin_floating_models, n_in_list):
    (x_train, y_train), (x_test, y_test) = data_generator(dangjin_data, n_in, n_out, ratio, x_name, y_name)
    model.fit(x_train, y_train, eval_set = [(x_test, y_test)], early_stopping_rounds = esr)
    del x_train, x_test, y_train, y_test

# dangjin_models training
x_name = ["temp_obs","ws_obs","humid_obs","dangjin"]
y_name = ["dangjin"]
dangjin_floating_models = [build_xgb(params) for n_in in n_in_list]

for model, n_in in zip(dangjin_floating_models, n_in_list):
    (x_train, y_train), (x_test, y_test) = data_generator(dangjin_data, n_in, n_out, ratio, x_name, y_name)
    model.fit(x_train, y_train, eval_set = [(x_test, y_test)], early_stopping_rounds = esr)
    del x_train, x_test, y_train, y_test


'''

# comparing result
def submission_predict(model, x_data, n_predict, model_type = "uv"):

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
    

    if model_type == "uv":
        x_data_after = x_data
        for i in range(n_predict):
            y_pred = model.predict(x_data_after)
            x_data_after = np.append(x_data_after, y_pred)[1:].reshape(1,-1)
            y_preds.append(y_pred)
        
        total_prediction = np.array(y_preds).reshape(-1,1)

    elif model_type == "mv":
        obs_data, fcst_data = x_data
        x_data_after = obs_data
        nf = fcst_data.shape[1] + 1
        for i in range(n_predict):
            y_pred = model.predict(x_data_after)
            x_data_next = np.append(fcst_data[i].reshape(1,-1), y_pred).reshape(1,-1)
            x_data_after = np.append(x_data_after, x_data_next)[nf:].reshape(1, -1)
            y_preds.append(y_pred)
        
        total_prediction = np.array(y_preds).reshape(-1,1)

    return total_prediction

# ulsan prediction
x_name = ["temp_obs","ws_obs","humid_obs","ulsan"]
y_name = ["ulsan"]
yhats_ulsan = None

term_3d = range(0, 24 * 3)
term_7d = range(0, 24 * 7)
term_30d = range(0, 24 * 30)

for n_in, model in zip(n_in_list, ulsan_models):
    name = "ulsan, timesteps: " + str(n_in / 24) + "-day"
    d_obs = ulsan_data[x_name][-24*30*1 - n_in : - 24*30*1].values.reshape(1,-1)
    d_fcst = ulsan_data[x_name[0:-1]][-24*30*1: ].values.reshape(24*30, len(x_name[0:-1]))
    prediction = submission_predict(model, (d_obs, d_fcst), model_type = "mv", n_predict = 24 * 30)
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
    
    
''' 

# ulsan
for n_in, model in zip(n_in_list, ulsan_humid):

    name = "humid_obs"
    input_prediction = ulsan_data[name][-24*30*1 - n_in : - 24*30*1].values.reshape(1,-1)
    prediction = submission_predict(model, input_prediction, n_predict = 24 * 30)
    yhat = prediction.reshape(-1, 1)

    if yhats_ulsan is None:
        yhats_ulsan = yhat
    else:
        yhats_ulsan = np.concatenate((yhats_ulsan, yhat), axis = 1)
    
    label = name + " - timesteps: " + str(n_in)

    for i, term in enumerate([term_3d, term_7d, term_30d]):
        plt.figure(i+1, figsize = (20, 10))
        plt.plot(yreal[term], label = "real")
        plt.plot(yhat[term], label = label)
        plt.ylabel(name)
        plt.title("ulsan humid with timesteps: %d"%(n_in))
        plt.legend()
        plt.show()
        
        
# Dangjin
for n_in, model in zip(n_in_list, dangjin_humid):

    name = "humid_obs"
    input_prediction = dangjin_data[name][-24*30*1 - n_in : - 24*30*1].values.reshape(1,-1)
    prediction = submission_predict(model, input_prediction, n_predict = 24 * 30)
    yhat = prediction.reshape(-1, 1)

    if yhats_ulsan is None:
        yhats_ulsan = yhat
    else:
        yhats_ulsan = np.concatenate((yhats_ulsan, yhat), axis = 1)
    
    label = name + " - timesteps: " + str(n_in)

    for i, term in enumerate([term_3d, term_7d, term_30d]):
        plt.figure(i+1, figsize = (20, 10))
        plt.plot(yreal[term], label = "real")
        plt.plot(yhat[term], label = label)
        plt.ylabel(name)
        plt.title("dangjin humid with timesteps: %d"%(n_in))
        plt.legend()
        plt.show()
        
'''

    

# weigted average method

def ensemble_weights(yhats, yreal):

    if not isinstance(yhats, np.ndarray):
        ValueError("yhats type error: must be np.ndarray")
        return None
    if not isinstance(yreal, np.ndarray):
        ValueError("yreal type error: must be np.array")
        return None        
    else:
        pass
    
    yreal = np.reshape(yreal, (-1, 1))
    err_matrix = yhats - yreal 

    C = np.zeros((len(ulsan_models), len(ulsan_models)), dtype = np.float32)
    
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            C[i,j] = np.dot(err_matrix[i,:].reshape(1,-1), err_matrix[j,:].reshape(-1,1)) / err_matrix.shape[1]
    
    w = np.zeros(len(ulsan_models))
    
    reverse_sum_C = 0
    for i in range(C.shape[0]):
        temp = np.sum(1 / (C[i,:] + 1e-6))
        reverse_sum_C += temp

    for i in range(len(ulsan_models)):
        w[i] = np.sum(1 / (C[i,:] + 1e-6)) / reverse_sum_C

    w = w.reshape(-1,1)

    return w

w_ulsan = ensemble_weights(yhats_ulsan, yreal)

y_weighted_sum = np.dot(yhats_ulsan, w_ulsan).reshape(-1,1)

# plot the ensemble model
term = term_30d
plt.plot(yreal[term], label = "real")
plt.plot(y_weighted_sum[term], label = "weighted sum")
plt.ylabel("ulsan, unit:None")
plt.title("Real and ensemble")
plt.legend()
plt.show()




'''

# ensemble model
# test: for uv model

# generate datasets from uv_models(uv_ulsan)

def ensemble_data_generate(models = uv_ulsan, data = ulsan_data, name = "ulsan", n_in_list = n_in_list):

    for i, n_in in enumerate(n_in_list):
        



# ensemble model training

n_models = len(uv_models)
timesteps_fcst = 24 * 30

W = np.random.randn(n_models, timesteps_fcst)
b = np.random.randn(timesteps_fcst)

class WeightSum:
    def __init__(self, W, b):
        self.n_features = n_features
        self.params = [W,b]
        self.x = None
        self.grads = [np.zeros_like(W), np.zeros_like(b)]

    def forward(self, x):

        W, b = self.params 
        out = np.matmul(x,W) + b
        self.x = x

        return out

    def backward(self, dout):
        W,b = self.params
        dx = np.matmul(dout, W.T)
        dW = np.matmul(self.x.T, dout)
        db = np.sum(dout, axis = 0)

        self.grads[0][...] = dW
        self.grads[1][...] = db

        return dx

class SGD:
    def __init__(self, lr = 0.01):
        self.lr = lr
    def update(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i] 

def custom_loss(x, t):
    
    h = tf.keras.losses.Huber(reduction = tf.keras.losses.Reduction.SUM)
    loss = h(x, t).numpy()

    from sklearn.metrics import mean_squared_error
    loss = mean_squared_error(x,t, squared = True)

    return loss

def custom_loss_back(x,t):

    dloss = 1/len(x) * (x-t).reshape(1,-1)

    return dloss

class ensemble_loss():
    def __init__(self):
        self.params = []
        self.grads = []
        self.out = None

    def forward(self,x, t):
        self.x = x
        self.t = t
        loss = custom_loss(x,t)
        self.loss = loss
        return loss

    def backward(self, dout = 1):
        x = self.x
        t = self.t
        dx = custom_loss_back(x,t)
        dx *= dout

        return dx
        
class ensemble():
    def __init__(self, n_models, timesteps_fcst):
        W = np.random.randn(n_models, timesteps_fcst)
        b = np.zeros(timesteps_fcst)

        self.layers = [
            WeightSum(W,b),
            ensemble_loss()
        ]

        self.params = [layer.params for layer in self.layers]
        self.grads = [layer.grads for layer in self.layers]

    def forward(self, x, t):

        for layer in self.layers:
            x = layer.forward(x)
        loss = x
        return loss

    def backward(self, dout = 1):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        
        return dout

    def predict(self, x):
        y_pred = self.layers[0].forward(x)

        return y_pred

optimizer = SGD(lr = 0.01)

ensemble_model = ensemble(n_models, timesteps_fcst)

def ensemble_train(x_data, y_data, ensemble, optimizer, max_iters = 100):

    data_size = x_data.shape[0]
    idx = np.random.permutation(data_size)
    x = x_data[idx]
    y = y_data[idx]

    loss_list = []

    batch_size = int(data_size / max_iters)

    for iters in range(max_iters):
        x_batch = x[iters*batch_size:(iters+1)*batch_size]
        y_batch = y[iters*batch_size:(iters+1)*batch_size]

        loss = ensemble.forward(x_batch, y_batch)
        ensemble.backward()
        optimizer.update(ensemble.params, ensemble.grads)

        total_loss += loss
        loss_count += 1

        if (iters +1) % 10 == 0:
            avg_loss = total_loss / loss_count
            print("|iters: %d | loss %.2f" %(iters +1, avg_loss))
            loss_list.append(avg_loss)
            total_loss, loss_count = 0 , 0

    return loss_list


ensemble_loss = ensemble_train( , , ensemble_model, optimizer, max_iters = 1000)

'''