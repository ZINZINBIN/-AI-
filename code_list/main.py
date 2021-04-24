# ======================================================================== #
# ========================== From gitlab Clone =========================== #
# ======================================================================== #

'''
- 수정 시각: 21.04.16 19:00
- 내용
1) gitlab clone시 작동할 수 있도록 수정
2) dangjin, ulsan data에 대한 xgboost model gridsearch 진행
'''

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

'''
- 일자, 시각 등은 'time' 이라는 변수명으로 통일
- 그 외의 변수명은 '변수명_파일명' 으로 지정(ulsan, dangjin)
- 가정하길 ulsan 데이터와 dangjin 데이터는 서로 무관하다. 
- 이후에 NAN 데이터는 dropna()를 이용해 모두 소거 예정
- 하나의 데이터셋을 구축한 뒤, 상관관게 분석 + ADF test 등을 진행
'''

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

energy_data_time_tmp = energy_data["time"].copy()

for i in range(energy_data.shape[0]):
    if energy_data["time"][i][-8:] == "24:00:00":
        energy_data["time"][i] = energy_data_time_tmp[i].replace("24:00:00", " 00:00:00")

    energy_data["time"][i] = pd.Timestamp(energy_data["time"][i])

energy_data = energy_data.astype({"time":"object"})

# 전 데이터 NAN 처리

dangjin_fcst_data = dangjin_fcst_data.dropna()
dangjin_obs_data = dangjin_obs_data.dropna()
energy_data = energy_data.dropna()
ulsan_fcst_data = ulsan_fcst_data.dropna()
ulsan_obs_data = ulsan_obs_data.dropna()

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

# total dataset 구성

from functools import reduce

list_dangjin = [dangjin_fcst_data, dangjin_obs_data, energy_data[["time","dangjin_floating","dangjin_warehouse","dangjin"]].copy()]
list_ulsan = [ulsan_fcst_data, ulsan_obs_data, energy_data[["time","ulsan"]].copy()]

dangjin_data = reduce(lambda  left,right: pd.merge(left, right, on=['time'], how='inner'), list_dangjin)
ulsan_data = reduce(lambda  left,right: pd.merge(left, right, on=['time'], how='inner'), list_ulsan)

display(dangjin_data)
display(ulsan_data)

'''
- dangjin 및 ulsan 지역 내 모든 측정/예측 변수를 하나의 데이터셋에 포함
- 중복되는 시간대에 대해서만 선별 후 하나의 데이터셋으로 통합
'''

# EDA 
import seaborn as sns
from scipy import stats

# stats.pearsonr을 이용해 각 변수간 pearson correlation을 구한다. 
# 이후 dangjin과 ulsan datasets에 대한 heatmap을 그린다. 

var_names = dangjin_data.copy().drop(columns = ["dangjin_floating","dangjin_warehouse", "dangjin", "time"]).columns

print("statistical analysis with dangjin_floating: pearson correlation coefficient")
print("===========================================================================")
for var_name in var_names:
    r, p = stats.pearsonr(dangjin_data[var_name], dangjin_data["dangjin_floating"])
    log = "variable: %s, pearson's r : %5f, p-value: %5f"
    fmt = log %(var_name, r,p)
    print(fmt)
print("===========================================================================\n")
print("statistical analysis with dangjin_warehouse: pearson correlation coefficient")
print("===========================================================================")
for var_name in var_names:
    r, p = stats.pearsonr(dangjin_data[var_name], dangjin_data["dangjin_warehouse"])
    log = "variable: %s, pearson's r : %5f, p-value: %5f"
    fmt = log %(var_name, r,p)
    print(fmt)
print("===========================================================================\n")
print("statistical analysis with dangjin: pearson correlation coefficient")
print("===========================================================================")
for var_name in var_names:
    r, p = stats.pearsonr(dangjin_data[var_name], dangjin_data["dangjin"])
    log = "variable: %s, pearson's r : %5f, p-value: %5f"
    fmt = log %(var_name, r,p)
    print(fmt)
print("===========================================================================\n")

dangjin_data.corr("pearson")
sns.heatmap(dangjin_data.corr("pearson"))

# ulsan data

var_names = ulsan_data.copy().drop(columns = ["ulsan", "time"]).columns

print("statistical analysis with ulsan: pearson correlation coefficient")
print("===========================================================================")
for var_name in var_names:
    r, p = stats.pearsonr(ulsan_data[var_name], ulsan_data["ulsan"])
    log = "variable: %s, pearson's r : %5f, p-value: %5f"
    fmt = log %(var_name, r,p)
    print(fmt)
print("===========================================================================\n")

ulsan_data.corr("pearson")
sns.heatmap(ulsan_data.corr("pearson"))

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
        #cols.append(data_copy[x_name].shift(i)) # col: [data_copy.shift(n_in), .... data_copy.shift(1)]
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

'''
- IDE 환경에서는 !sudo pip install 작동 X(Jupyter는 된다)
- dangjin, ulsan 두 지역에 대한 발전량 예측 알고리즘
- gridsearchCV를 이용한 parameter tunning 시도
'''

# dangjin
x_name = ["temp_obs","ws_obs","humid_obs","dangjin_floating", "dangjin_warehouse", "dangjin"]
y_name = ["temp_obs","ws_obs","humid_obs","dangjin_floating", "dangjin_warehouse", "dangjin"]
n_features = len(x_name)
n_in = 24 * 3
n_out = 1
ratio = 0.8 # train size ratio

dangjin_data_supervised = series_to_supervised(dangjin_data, x_name, y_name, n_in, n_out, dropnan = True)

x_data = dangjin_data_supervised.values[:,:-6]
y_data = dangjin_data_supervised.values[:,-6:]

# train_test split
data_size = x_data.shape[0]
train_size = int(data_size * ratio)

x_train = x_data[0:train_size]
x_test = x_data[train_size:]

y_train = y_data[0:train_size]
y_test = y_data[train_size:]

# build model
params = {
    "booster":"dart",
    "objective":"reg:squarederror",
    "learning_rate":0.001,
    "max_depth":15,
    "n_estimators":1000,
    "nthread":-1,
    "gamma":10.0,
    "subsample":0.8,
    "colsample_bytree":0.8,
    "colsample_bylevel":0.8,
    "min_child_weight":3,
    "reg_lambda":1.0,
    "reg_alpha":0.1
}

'''
# gridsearch
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline

param_grid = {
    "eta":[0.001, 0.01, 0.1, 0.3],
    "booster":["gbtree", "linear"],
    "objective":["reg:squarederror", "reg:linear"],
    "learning_rate":[0.01, 0.001],
    "max_depth":[3, 5, 10, 15, 20],
    "n_estimators":[50, 100, 200, 500, 1000],
    "nthread":[-1],
    "gamma":[0, 0.01, 0.1, 1.0, 10, 100],
    "subsample":[0.2, 0.5, 0.8],
    "colsample_bytree":[0.2, 0.5, 0.8],
    "colsample_bylevel":[0.2, 0.5, 0.8],
    "min_child_weight":[1,3,5],
    "reg_lambda":[0.01, 0.1, 1.0, 10],
    "reg_alpha":[0, 0.1, 1.0]
}

cv = KFold(n_splits = 5, random_state = 42)
model = xgb.XGBRegressor()

grid = GridSearchCV(model, param_grid = param_grid, cv = cv, scoring = "neg_root_mean_squared_error", n_jobs = -1, refit = True)

# model_params for dangjin_floating
grid.fit(x_train, y_train[:,3].reshape(-1,1))
print("optimized parameters for dangjin_floating: ",grid.best_params_)

# model_params for dangjin_warehouse
grid.fit(x_train, y_train[:,4].reshape(-1,1))
print("optimized parameters for dangjin_warehouse: ",grid.best_params_)

# model_params for dangjin
grid.fit(x_train, y_train[:,5].reshape(-1,1))
print("optimized parameters for dangjin: ",grid.best_params_)

'''


# build model and parameter setting

def build_xgb(params):
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

dangjin_models = [build_xgb(params) for i in range(n_features)]

# train model
for i in range(n_features):
    var_name_predict = y_name[i]
    y_var_train = y_train[:,i].reshape(-1,1)
    y_var_test = y_test[:,i].reshape(-1,1)
    print("forecasting variable:%s"%(var_name_predict))
    dangjin_models[i].fit(x_train, y_var_train, early_stopping_rounds = 400, eval_set = [(x_test, y_var_test)])

# submission_prediction

def submission_predict(models, x_data, n_predict):
    '''
    - for multi-varaible prediction
    - x_data:(timesteps, n_features)
    - n_predict: timesteps for forecasting
    - x_data_after: x_data[1:] + predict_value
    '''
    total_prediction = None
    y_preds = []
    x_data_after = x_data

    for i in range(n_predict):
        y_preds = []
        for j in range(n_features):
            y_pred = models[j].predict(x_data_after)
            y_preds.append(y_pred)

        y_preds = np.array(y_preds).reshape(1,-1)
        x_data_after = np.concatenate((x_data_after, y_preds), axis = 1)[0,n_features:].reshape(1,-1)
        

        if total_prediction is None:
            total_prediction = y_preds
        else:
            total_prediction = np.concatenate((total_prediction, y_preds), axis = 0)
    
    total_prediction = total_prediction.reshape(-1, n_features)

    return total_prediction

# test
input_prediction = dangjin_data[x_name][-24*30*1 - n_in:-24*30*1].values.reshape(1,-1)
actual = dangjin_data[y_name][-24*30*1:].values
prediction_dangjin = submission_predict(dangjin_models, input_prediction, n_predict = 24 * 30)

for i in range(len(y_name)):
    var_name = y_name[i]
    plt.figure(i)  
    
    yreal = actual[:,i]
    yhat = prediction_dangjin[:,i]
    plt.title(var_name + "prediction")
    plt.ylabel(var_name)
    plt.plot(yreal, "r", label = "actual value")
    plt.plot(yhat, "b", label = "forecasting")
    plt.legend()
    plt.show()

# ulsan
x_name = ["temp_obs","ws_obs","humid_obs","ulsan"]
y_name = ["temp_obs","ws_obs","humid_obs","ulsan"]
n_features = len(x_name)
n_in = 24 * 3
n_out = 1
ratio = 0.8 # train size ratio

ulsan_data_supervised = series_to_supervised(ulsan_data, x_name, y_name, n_in, n_out, dropnan = True)

x_data = ulsan_data_supervised.values[:,:-4]
y_data = ulsan_data_supervised.values[:,-4:]

# train_test split
data_size = x_data.shape[0]
train_size = int(data_size * ratio)

x_train = x_data[0:train_size]
x_test = x_data[train_size:]

y_train = y_data[0:train_size]
y_test = y_data[train_size:]

# build model

params = {
    "booster":"dart",
    "objective":"reg:squarederror",
    "learning_rate":0.001,
    "max_depth":15,
    "n_estimators":1000,
    "nthread":-1,
    "gamma":10.0,
    "subsample":0.8,
    "colsample_bytree":0.8,
    "colsample_bylevel":0.8,
    "min_child_weight":3,
    "reg_lambda":1.0,
    "reg_alpha":0.1
}

ulsan_models = [build_xgb(params) for i in range(n_features)]

# train model
for i in range(n_features):
    var_name_predict = y_name[i]
    y_var_train = y_train[:,i].reshape(-1,1)
    y_var_test = y_test[:,i].reshape(-1,1)
    print("forecasting variable:%s"%(var_name_predict))
    ulsan_models[i].fit(x_train, y_var_train, early_stopping_rounds = 400, eval_set = [(x_test, y_var_test)])


# submission_prediction for ulsan
# test
input_prediction = ulsan_data[x_name][-24*30*1 - n_in:-24*30*1].values.reshape(1,-1)
actual = ulsan_data[y_name][-24*30*1:].values
prediction_ulsan = submission_predict(ulsan_models, input_prediction, n_predict = 24 * 30)

for i in range(len(y_name)):
    var_name = y_name[i]
    plt.figure(i)  
    
    yreal = actual[:,i]
    yhat = prediction_ulsan[:,i]
    plt.title(var_name + "prediction")
    plt.ylabel(var_name)
    plt.plot(yreal, "r", label = "actual value")
    plt.plot(yhat, "b", label = "forecasting")
    plt.legend()
    plt.show()

# NMAE-10 지표 함수

def sola_nmae(answer_df, submission_df):
    submission = submission_df[submission_df['time'].isin(answer_df['time'])]
    submission.index = range(submission.shape[0])
    
    # 시간대별 총 발전량
    sum_submission = submission.iloc[:,1:].sum(axis=1)
    sum_answer = answer_df.iloc[:,1:].sum(axis=1)
    
    # 발전소 발전용량
    capacity = {
        'dangjin_floating':1000, # 당진수상태양광 발전용량
        'dangjin_warehouse':700, # 당진자재창고태양광 발전용량
        'dangjin':1000, # 당진태양광 발전용량
        'ulsan':500 # 울산태양광 발전용량
    }
    
    # 총 발전용량
    total_capacity = np.sum(list(capacity.values()))
    
    # 총 발전용량 절대오차
    absolute_error = (sum_answer - sum_submission).abs()
    
    # 발전용량으로 정규화
    absolute_error /= total_capacity
    
    # 총 발전용량의 10% 이상 발전한 데이터 인덱스 추출
    target_idx = sum_answer[sum_answer>=total_capacity*0.1].index
    
    # NMAE(%)
    nmae = 100 * absolute_error[target_idx].mean()
    
    return nmae
