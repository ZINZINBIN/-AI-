# library
import os
import numpy as np
import pandas as pd
import scipy as sp
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import glob
import pickle
from lib_func_svm import *

# ======================================================================== #
# ===================== load data and preprocessing ====================== #
# ======================================================================== #

# load data
with open('./witt_preprocessing/pickles/dangjin_data.pkl','rb') as f:
    dangjin_data = pickle.load(f)
with open('./witt_preprocessing/pickles/ulsan_data.pkl','rb') as f:
    ulsan_data = pickle.load(f)

dangjin_data_ = dangjin_data.copy()
ulsan_data_ = ulsan_data.copy()

'''
# energy = 0 drop
dangjin_data = dangjin_data[dangjin_data["dangjin"] != 0]
dangjin_data = dangjin_data[dangjin_data["dangjin_floating"] != 0]
dangjin_data = dangjin_data[dangjin_data["dangjin_warehouse"] != 0]
ulsan_data = ulsan_data[ulsan_data["ulsan"] != 0]
'''

# ======================================================================== #
# ================= build model and training(SVM model) ================== #
# ======================================================================== #

# model architecture
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

params = {
    "kernel":"poly",
    "degree":7,
    "gamma":"scale",
    "tol":1e-5,
    "C":10.0,
    "epsilon":0.001,}

# ======================================================================== #
# ================= ulsan model training and evaluation ================== #
# ======================================================================== #

# ulsan 

model_num = 10

x_name = ["Day_cos","Day_sin", "Year_cos","Year_sin", "Temperature", "Humidity", "Wind_X", "Wind_Y", "Cloud"]
y_name = ["ulsan"]

'''
ulsan_supervised = series_to_supervised(ulsan_data, x_name, y_name, n_in = 24, n_out = 1, dropnan = True)
x, y = ulsan_supervised.iloc[:,:-1].values, ulsan_supervised.iloc[:,-1].values

x_train, x_val, y_train, y_val = train_test_split(x,y, test_size = 0.2, random_state = 42)
'''


x_train, x_val, y_train, y_val = data_generate_svm(ulsan_data.iloc[0:-24*30], x_name, y_name, test_size = 0.1)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)


x_train_split, y_train_split = data_split(x_train, y_train, n = model_num)
x_val_split, y_val_split = data_split(x_val, y_val, n = model_num)

ulsan_model_list = [build_svm(params) for i in range(model_num)]
ulsan_model_nmae_avg = 0

# model training
for i, model in enumerate(ulsan_model_list):
    train_x, train_y = x_train_split[i], y_train_split[i]
    val_x, val_y = x_val_split[i], y_val_split[i]
    model.fit(train_x, train_y)

    pred_y = model.predict(val_x).reshape(-1,1)
    nmae = sola_nmae(val_y, pred_y)
    ulsan_model_nmae_avg += nmae
        
    print("ulsan_model %d nmae: %.2f"%(i, nmae))
 
    
ulsan_model_nmae_avg /= model_num
print("ulsan_model average nmae: ", ulsan_model_nmae_avg)


# ensemble
x_ens = ulsan_data_[x_name].iloc[-24*30:].values.reshape(-1, len(x_name))
y_ens = ulsan_data_[y_name].iloc[-24*30:].values.reshape(-1,1)

ulsan_yhats = []

for i, model in enumerate(ulsan_model_list):

    yhat = model.predict(x_ens).reshape(-1,1)
    ulsan_yhats.append(yhat)
    
ulsan_yhats = np.array(ulsan_yhats).reshape(-1, model_num)
ulsan_weights = ensemble_weights(ulsan_yhats, y_ens, model_num)

ulsan_ensemble_prediction = np.dot(ulsan_yhats , ulsan_weights)
ulsan_ensemble_nmae = sola_nmae(y_ens, ulsan_ensemble_prediction, cap = "ulsan")

print("ulsan_ensemble_nmae for 21.01 data: ", ulsan_ensemble_nmae)
