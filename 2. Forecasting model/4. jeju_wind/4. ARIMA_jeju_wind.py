import numpy as np
import math
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

# MAPE function
def MAPE(y_test, y_pred):
    return np.mean(np.abs((y_test - y_pred) / y_test)) * 100

# MAE function
def MAE(y_test, y_pred):
    return np.mean(np.abs(y_test - y_pred))


# Convert an array of values into a dataset matrix → Sliding Window
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


# Fix random seed
np.random.seed(42)

# Parameter set
nInput = 1 # Output shape
nOutput = 1 # Input shape
look_back = 23 # https://youtu.be/Say55dyAwx0
nStep = 24 # Forecasting step
nFeature = 1 # Features, columns
shift = 24 # move data

# Make empty data set
txt_r2_test = []
txt_pred_test_x = []
txt_rmse_test = []
txt_mape_test = []
txt_mae_test = []

# Load data
filename = 'sampling_jeju_wind_50.csv'
Data = pd.read_csv(filename, header=None)
V = Data.values

# Choose data (Can remove this)
V = V[:1]
index = Data.index
values = []

for i in range(V.shape[0]):
    # Many datasets loopable
    values = V[i]
    values = np.reshape(values, (-1, 1))
    values = values[:2000]

    # Split data
    split_point = int(len(values) * 0.8)
    train_set = values[:split_point]
    trainL = len(train_set)
    test_set = values[split_point:]
    testL = len(test_set)

    # Scale
    rs = RobustScaler() # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html
    values = rs.fit_transform(values)

    # Flatten
    train_set = train_set.reshape(-1)
    test_set = test_set.reshape(-1)

    # Make data set
    trainX = values[: len(train_set) - shift]
    train_y = values[look_back + shift: split_point]
    train_x, YY = createTrainData(trainX, look_back)
    testX = values[len(train_set) - look_back - shift: len(train_set) + len(test_set) - shift]
    test_y = values[split_point:]
    test_x, YY = createTrainData(testX, look_back)

    # ARIMA code, order = (p, d, q)
    test_history = [x for x in test_y]
    testPredict = list()
    for t in range(len(test_y)):
        model = ARIMA(test_history, order=(16, 1, 1))
        model_fit = model.fit()
        output = model_fit.forecast(steps=nStep)
        yhat = output[-1]
        testPredict.append(yhat)
        obs = test_y[t]
        test_history.append(obs)
        print('predicted=%f, expected=%f' % (yhat, obs))

    # Reshape for inverse scale
    testPredict = np.reshape(np.array(testPredict), (-1, 1))
    test_y = np.reshape(np.array(test_y), (-1, 1))

    # Inverse scale
    testPredict = rs.inverse_transform(testPredict)
    test_y = rs.inverse_transform(test_y)

    # Evaluation model result data
    r2_test = r2_score(test_y, testPredict)
    rmse_test = math.sqrt(mean_squared_error(test_y, testPredict))
    mae_test = MAE(test_y, testPredict)
    mape_test = MAPE(test_y, testPredict)

    # Append evaluation score
    txt_r2_test.append(r2_test)
    txt_rmse_test.append(rmse_test)
    txt_mae_test.append(mae_test)
    txt_mape_test.append(mape_test)
    txt_pred_test_x.append(testPredict)

# Reshape for save data frame
txt_r2_test = np.reshape(np.array(txt_r2_test), (V.shape[0], -1))
txt_pred_test_x = np.reshape(np.array(txt_pred_test_x), (V.shape[0], -1))
txt_rmse_test = np.reshape(np.array(txt_rmse_test), (V.shape[0], -1))
txt_mae_test = np.reshape(np.array(txt_mae_test), (V.shape[0], -1))
txt_mape_test = np.reshape(np.array(txt_mape_test), (V.shape[0], -1))

# Save evaluation CSV file
ind = ['r2_test', 'rmse_test', 'mae_test', 'mape_test']
df1 = np.concatenate((txt_r2_test, txt_rmse_test, txt_mae_test, txt_mape_test), axis=1)
df1 = df1.T
df1 = pd.DataFrame(df1, index=ind)
df1.to_csv('.\\result/ARIMA_jeju_wind_performance.csv')