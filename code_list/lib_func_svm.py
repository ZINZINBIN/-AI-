import numpy as np
import pandas as pd
import scipy as sp
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import pickle

# Lagrangian interpolation

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


# supervised learning을 위한 preprocessing

def series_to_supervised(data, x_name, y_name, n_in, n_out, dropnan = False):

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

from sklearn.model_selection import train_test_split

def data_generate_svm(data, x_name, y_name, n_out = 1, test_size = 0.3):
    
    x = data.copy()[x_name].values.reshape(-1, len(x_name))
    y = data.copy()[y_name].values.reshape(-1, len(y_name))
    
    x_train, x_val, y_train, y_val = train_test_split(x,y,test_size = test_size, random_state = 42)
    
    return x_train, x_val, y_train, y_val

def data_split(x,y, n = 10):
    
    x_split = []
    y_split = []
    
    total_len = x.shape[0]
    idx = np.arange(0, total_len)
    np.random.shuffle(idx)
    
    unit_len = int(total_len / n)
    
    for i in range(n):
        unit_idx = idx[unit_len * i : unit_len * (i+1)]
        unit_x = x[unit_idx]
        unit_y = y[unit_idx]
        
        x_split.append(unit_x)
        y_split.append(unit_y)
        
    return x_split, y_split

# build model
# build model and parameter setting
    
from sklearn.svm import SVR    

def build_svm(params = None):
    
    if params is None:
        model = SVR()
    else:
        kernel = params["kernel"]
        degree = params["degree"]
        gamma = params["gamma"]
        C = params["C"]
        epsilon = params["epsilon"]
        tol = params["tol"]
        
        model = SVR(kernel = kernel, degree = degree, gamma = gamma, C = C, epsilon = epsilon, tol = tol)
        
    return model

# comparing result
def submission_predict_svm(model, n_predict, fcst_data):
    
    y_preds = model.predict(fcst_data)
    
    if type(y_preds) != np.ndarray:
        y_preds = np.array([v for v in y_preds]).reshape(-1,1)
    else:
        y_preds = y_preds.reshape(-1,1)

    return y_preds



# weight sum 
def ensemble_weights(yhats, yreal, model_num):

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

    C = np.zeros((model_num, model_num), dtype = np.float32)
    
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            C[i,j] = np.dot(err_matrix[i,:].reshape(1,-1), err_matrix[j,:].reshape(-1,1)) / err_matrix.shape[1]
    
    w = np.zeros(model_num)
    
    reverse_sum_C = 0
    for i in range(C.shape[0]):
        temp = 1 / (C[i,i] + 1e-6)
        reverse_sum_C += temp

    for i in range(model_num):
        w[i] = 1 / (C[i,i] + 1e-6) / reverse_sum_C

    w = w.reshape(-1,1)

    return w

# NMAE-10 지표 함수

capacity = {
    'dangjin_floating':1000, # 당진수상태양광 발전용량
    'dangjin_warehouse':700, # 당진자재창고태양광 발전용량
    'dangjin':1000, # 당진태양광 발전용량
    'ulsan':500 # 울산태양광 발전용량
}

def sola_nmae(y_true, y_pred, cap = "ulsan"):    
    abs_err = np.abs(y_true - y_pred)
    abs_err /= capacity[cap]
    target_idx = np.where(y_pred >= capacity[cap] * 0.1)
    result = 100 * abs_err[target_idx].mean()
    return result



# submission obs data preprocessing
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

def day_of_year(datetime): #pd.datetime
    return pd.Period(datetime, freq='D').dayofyear

def add_seasonality(df):
    new_df = df.copy()
    
    new_df['Day_cos'] = new_df['time'].apply(lambda x: np.cos(x.hour * (2 * np.pi) / 24))
    new_df['Day_sin'] = new_df['time'].apply(lambda x: np.sin(x.hour * (2 * np.pi) / 24))

    new_df['Year_cos'] = new_df['time'].apply(lambda x: np.cos(day_of_year(x) * (2 * np.pi) / 365))
    new_df['Year_sin'] = new_df['time'].apply(lambda x: np.sin(day_of_year(x) * (2 * np.pi) / 365))

    return new_df

# obs_preprocessing

def obs_preprocessing(df):

    df_obs = df.copy()
    df_obs.rename(
        columns = {
                "일시":"time",
                "기온(°C)":"Temperature",
                "풍속(m/s)":"WindSpeed",
                "풍향(16방위)":"WindDirection",
                "습도(%)":"Humidity",
                "전운량(10분위)":"Cloud"
            }, inplace = True)

    df_obs.drop(columns = ["지점", "지점명"])

    df_obs = df_obs.join(preprocess_wind(df_obs))

    for i in range(df_obs.shape[0]):
        df_obs["time"][i] = pd.to_datetime(df_obs["time"][i])
    
    df_obs = df_obs.astype({"time":"object"})
    df_obs = add_seasonality(df_obs)
        
    return df_obs
