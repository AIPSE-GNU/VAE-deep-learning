# cnn model
from numpy import mean
from numpy import std
import numpy as np
from numpy import dstack
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split
# from keras.utils import to_categorical

# 데이터 호출 (sampling의 경우 test data는 사용되지 않음)
from ex_load_data_002 import load_dataset
x_train = load_dataset()

# 데이터 스케일링
scaler = MinMaxScaler(feature_range=(0, 1))
x_train = scaler.fit_transform(x_train.reshape(x_train.shape[0], -1)).reshape(x_train.shape)

import os
from util.loaders import load_mnist
from models.VAE_CONV_D1 import VariationalAutoencoder


# 실행 매개변수
SECTION = 'vae_conv_d1'
RUN_ID = '0001'
DATA_NAME = 'res'
RUN_FOLDER = 'run/{}/'.format(SECTION)
RUN_FOLDER += '_'.join([RUN_ID, DATA_NAME])

if not os.path.exists(RUN_FOLDER):
    os.mkdir(RUN_FOLDER)
    os.mkdir(os.path.join(RUN_FOLDER, 'viz'))
    os.mkdir(os.path.join(RUN_FOLDER, 'images'))
    os.mkdir(os.path.join(RUN_FOLDER, 'weights'))

MODE = 'build'

# CONV_1D를 합성곱 신경망으로 사용하는 VAE, 압축 인자(z_dim)=# of extracted features
VAE = VariationalAutoencoder(
    input_dim = (24, 4)
    , encoder_conv_filters = [32, 64, 64, 64]
    , encoder_conv_kernel_size = [3, 3, 3, 3]
    # , encoder_conv_strides = [1,2,2,1]
    , encoder_conv_strides = [1, 1, 2, 2]
    , decoder_conv_t_filters = [64, 64, 32, 1]
    , decoder_conv_t_kernel_size = [3, 3, 3, 3]
    # , decoder_conv_t_strides = [1,2,2,1]
    , decoder_conv_t_strides=[1, 1, 2, 2]
    , z_dim = 2
)

if MODE == 'build':
    VAE.save(RUN_FOLDER)
else:
    VAE.load_weights(os.path.join(RUN_FOLDER, 'weights/weights.h5'))

# hyper-parameter 설정
LEARNING_RATE = 0.0005
R_LOSS_FACTOR = 1000

BATCH_SIZE = 32
EPOCHS = 500
PRINT_EVERY_N_BATCHES = 100
INITIAL_EPOCH = 0

VAE.compile(LEARNING_RATE, R_LOSS_FACTOR)

VAE.train(
    x_train
    , batch_size = BATCH_SIZE
    , epochs = EPOCHS
    , run_folder = RUN_FOLDER
    , print_every_n_batches = PRINT_EVERY_N_BATCHES
    , initial_epoch = INITIAL_EPOCH
)

