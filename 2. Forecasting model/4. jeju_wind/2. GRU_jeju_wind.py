import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import GRU
import math
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score
import warnings
from sklearn.metrics import mean_squared_error

# Evaluation function
def MAPE(y_test, y_pred):
   return np.mean(np.abs((y_test - y_pred) / y_test)) * 100
def MAE(y_test, y_pred):
   return np.mean(np.abs(y_test - y_pred))

warnings.filterwarnings('ignore')

# Convert an array of values into a dataset matrix → Sliding Window (Moving Window)
def createTrainData(xData, look_back):
    m = np.arange(len(xData) - look_back)
    x, y = [], []

    for i in m:
        a = xData[i:(i + look_back)]
        x.append(a)
    xBatch = np.reshape(np.array(x), (len(m), look_back, 1))

    for i in m + 1:
        a = xData[i:(i + look_back + 1)]
        y.append(a[-1])
    yBatch = np.reshape(np.array(y), (len(m), 1))

    return xBatch, yBatch

def loading_model(load_model=True): # True is loading weight, False is fitting and saving model
    if load_model:
        model.load_weights(f'./save_model/GRU/GRU_land_solar{i}.h5')
    else:
        # Trains the model for a given number of epochs (iterations on a dataset).
        model.fit(train_x, train_y, epochs=20, batch_size=18, validation_split= 0.2, verbose=2)
        model.save_weights(f"./save_model/GRU/GRU_land_solar{i}.h5")

# Fix random seed
np.random.seed(42)

# Parameter Set
nInput = 1  # Output shape
nOutput = 1  # Input shape
look_back = 23  # https://youtu.be/Say55dyAwx0
nStep = 1  # Forecasting step
nHidden = 32  # number of hidden layer
nFeature = 1  # Features, columns
shift = 24  # move data

# Make Empty Data Set
txt_r2_train = []
txt_r2_test = []
txt_Prediction = []
txt_pred_train_x = []
txt_pred_test_x = []
txt_rmse_train = []
txt_rmse_test = []
txt_mape_train = []
txt_mape_test = []
txt_mae_train = []
txt_mae_test = []

# Load Data
filename = 'sampling_jeju_wind_50.csv'
Data = pd.read_csv(filename, header=None)
V = Data.values

# Many datasets loopable
for i in range(V.shape[0]):
    print(i)
    values = V[i]
    values = np.reshape(values, (-1, 1))

    # Split Data
    split_point = int(len(values) * 0.8)
    train_set = values[:split_point]
    trainL = len(train_set)
    test_set = values[split_point:]
    testL = len(test_set)

    # Scale
    rs = RobustScaler() # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html
    train_set = rs.fit_transform(train_set)
    test_set = rs.transform(test_set)

    # Flatten
    train_set = train_set.reshape(-1)
    test_set = test_set.reshape(-1)

    # Moving window
    train_x, train_y = createTrainData(train_set, look_back)
    test_x, test_y = createTrainData(test_set, look_back)

    # Sequential model
    model=Sequential()
    model.add(GRU(nHidden, input_shape = (look_back, nFeature)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    loading_model()

    # Prediction data
    trainPredict = model.predict(train_x)
    testPredict = model.predict(test_x)

    # Forecasting Data
    lastData = np.copy(train_set)
    dx = np.copy(lastData)
    Predict = []
    result = values
    result = rs.transform(result)

    # Updatable forecasting loop
    for i in range(shift):
        result = result[-look_back:]
        result = result.reshape(-1, look_back, nFeature)
        a = model.predict(result)
        result = np.append([result], [a])
        Predict.append(a)

    # Reshape for inverse scale
    Predict = np.array(Predict)
    Predict = np.reshape(Predict, (-1, nFeature))

    # Inverse scale
    trainPredict = rs.inverse_transform(trainPredict)
    testPredict = rs.inverse_transform(testPredict)
    Predict = rs.inverse_transform(Predict)
    train_y = rs.inverse_transform(train_y)
    test_y = rs.inverse_transform(test_y)

    # Evaluation model result data
    r2_train = r2_score(train_y, trainPredict)
    r2_test = r2_score(test_y, testPredict)
    mse_train = mean_squared_error(train_y, trainPredict)
    mse_test = mean_squared_error(test_y, testPredict)
    rmse_train = math.sqrt(mse_train)
    rmse_test = math.sqrt(mse_test)
    mae_train = MAE(train_y, trainPredict)
    mae_test = MAE(test_y, testPredict)
    mape_train = MAPE(train_y, trainPredict)
    mape_test = MAPE(test_y, testPredict)

    # Returns the loss value & metrics values for the model in test mode.
    print('Train Score: %.2f MSE (%.2f RMSE)' % (mse_train, rmse_train))
    print('Test Score: %.2f MSE (%.2f RMSE)' % (mse_test, rmse_test))
    print('Train Score: %.2f r2_train' % r2_train)
    print('Test Score: %.2f r2_train' % r2_test)

    # Append evaluation score
    txt_r2_train.append(r2_train)
    txt_r2_test.append(r2_test)
    txt_rmse_train.append(rmse_train)
    txt_rmse_test.append(rmse_test)
    txt_mae_train.append(mae_train)
    txt_mae_test.append(mae_test)
    txt_mape_train.append(mape_train)
    txt_mape_test.append(mape_test)
    txt_pred_train_x.append(trainPredict)
    txt_pred_test_x.append(testPredict)
    txt_Prediction.append(Predict)

# Reshape for save data frame
txt_r2_train = np.reshape(np.array(txt_r2_train), (V.shape[0], -1))
txt_r2_test = np.reshape(np.array(txt_r2_test), (V.shape[0], -1))
txt_pred_train_x = np.reshape(np.array(txt_pred_train_x), (V.shape[0], -1))
txt_pred_test_x = np.reshape(np.array(txt_pred_test_x), (V.shape[0], -1))
txt_Prediction = np.reshape(np.array(txt_Prediction), (V.shape[0], -1))
txt_rmse_train = np.reshape(np.array(txt_rmse_train), (V.shape[0], -1))
txt_rmse_test = np.reshape(np.array(txt_rmse_test), (V.shape[0], -1))
txt_mae_train = np.reshape(np.array(txt_mae_train), (V.shape[0], -1))
txt_mae_test = np.reshape(np.array(txt_mae_test), (V.shape[0], -1))
txt_mape_train = np.reshape(np.array(txt_mape_train), (V.shape[0], -1))
txt_mape_test = np.reshape(np.array(txt_mape_test), (V.shape[0], -1))


# The empty cell created by the moving window
nv = np.empty((V.shape[0], look_back))
nv[:] = np.nan

# Save forecasting data file
df = np.concatenate((nv, txt_pred_train_x, nv, txt_pred_test_x, txt_Prediction), axis=1)
df = df.T
inde = pd.period_range(start='2030-01-01 01:00', end= None, periods=8784, freq='H')
df = pd.DataFrame(df, index=inde)
df.to_csv('.\\result\GRU_jeju_wind.csv')


# Save evaluation CSV File
ind = ['r2_train', 'r2_test', 'rmse_train', 'rmse_test', 'mae_train', 'mae_test', 'mape_train', 'mape_test']
df1 = np.concatenate((txt_r2_train, txt_r2_test, txt_rmse_train, txt_rmse_test, txt_mae_train, txt_mae_test, txt_mape_train, txt_mape_test), axis=1)
df1 = df1.T
df1 = pd.DataFrame(df1, index=ind)
df1.to_csv('.\\result\GRU_jeju_wind_performance.csv')