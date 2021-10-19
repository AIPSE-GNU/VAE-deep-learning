import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import norm
from util.loaders import load_mnist, load_model
from models.VAE_CONV_D1 import VariationalAutoencoder
import pandas as pd
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import tensorflow


# 실행 매개변수
SECTION = 'vae_conv_d1'
RUN_ID = '0001'
DATA_NAME = 'res'
RUN_FOLDER = 'run/{}/'.format(SECTION)
RUN_FOLDER += '_'.join([RUN_ID, DATA_NAME])

# 훈련된 모델이 저장된 폴더에서 훈련된 모델 호출
VAE = load_model(VariationalAutoencoder, RUN_FOLDER)


# 훈련에 사용된 scaler를 다시 호출하기 위한 프로세스
from ex_load_data_002 import load_dataset
x_train= load_dataset()
scaler = MinMaxScaler(feature_range=(0, 1))
x_train = scaler.fit_transform(x_train.reshape(x_train.shape[0], -1)).reshape(x_train.shape)
x_test= load_dataset()

# 생성하고자 하는 sample 개수(n_to_show) 설정
n_to_show = 365*50
znew = np.random.normal(size=(n_to_show, VAE.z_dim))
reconst_test = VAE.decoder.predict(np.array(znew))

# extracted features를 활용하여 생선한 sample들을 원래 구조로 전환(inverse_transform)
# 개별 파일로 분리하여 해당 폴더에 sample들을 저장
x_test = scaler.inverse_transform(reconst_test.reshape(reconst_test.shape[0], -1)).reshape(reconst_test.shape)
x_test_jeju_solar = x_test[:, :, 0]
x_test_jeju_wind = x_test[:, :, 1]
x_test_land_solar = x_test[:, :, 2]
x_test_land_wind = x_test[:, :, 3]

generative_test_jeju_solar = pd.DataFrame(x_test_jeju_solar)
generative_test_jeju_wind = pd.DataFrame(x_test_jeju_wind)
generative_test_land_solar = pd.DataFrame(x_test_land_solar)
generative_test_land_wind = pd.DataFrame(x_test_land_wind)

generative_test_jeju_solar.to_csv("./RES/vae_generative_test_jeju_solar_01.csv", header=False, index=False)
generative_test_jeju_wind.to_csv("./RES/vae_generative_test_jeju_wind_01.csv", header=False, index=False)
generative_test_land_solar.to_csv("./RES/vae_generative_test_land_solar_01.csv", header=False, index=False)
generative_test_land_wind.to_csv("./RES/vae_generative_test_land_wind_01.csv", header=False, index=False)