from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

# 호출 순서:
# load_dataset => load_dataset_group => load_group => load_file

# load_dataset:
# A. 최종 실행 함수 (input 인자는 따로 없음)
# B. 최상위 폴더(here, RES)와 dataset 구분된 폴더(here, train or test) 접속 실행 함수인 load_dataset_group 포함

# load_dataset_group:
# A. 최상위 폴더와 dataset 구분된 폴더 (here, train or test) 호출을 통하여 최하위 폴더(here, RES) 및 파일들(csv) 접속

# load_group:
# A. 최상위 폴더부터 최하위 폴더까지의 경로와 개별 파일(csv) 연결을 filepath 형태로 load_file의 input 인자로 넘김
# B. for 문을 통하여 각각의 개별 파일(csv)들에 대하여 load_file을 실행하고 결과 값들은 모두 append 한 후 dstack을 통하여 3차원 배열 형성

# load_file:
# A. 최상위 폴더부터 최하위 폴더까지의 경로와 개별 파일(csv)에 접속하여 실제 값들을 호출
# B. 이를 다시 위 함수들의 호출 순서 역으로 전파


# load a single file as a numpy array
# 개별 파일에 접속하여 값들을 배출
def load_file(filepath):
    dataframe = read_csv(filepath, header=None)
    return dataframe.values[1:, 1:]


# load a list of files and return as a 3d numpy array
# 전체 파일들 내의 개별 파일에 접속하여 호출된 값들을 리스트에 append 한 후 하나의 3차원 배열로 배출
def load_group(filenames, prefix=''):
    loaded = list()
    for name in filenames:
        data = load_file(prefix + name)
        loaded.append(data)
    # stack group so that features are the 3rd dimension
    # (sample 개수, 시간, feature)의 3차원 배열 형성
    loaded = dstack(loaded)
    return loaded


# load a dataset group, such as train or test
# 데이터 세트(train 혹은 test) 폴더에 대응되는 경로(filepath) 및 폴더 내 전체 파일들(filename) 이름을 load_group에 전달
def load_dataset_group(group, prefix=''):
    filepath = prefix + group + '/RES_YB/'

    filenames = list()
    # RE demand/solar
    filenames += ['row_data_jeju_solar(2030).csv', 'row_data_jeju_wind(2030).csv', 'row_data_land_solar(2030).csv', 'row_data_land_wind(2030).csv']
    # load input data
    X = load_group(filenames, filepath)
    return X


# load the dataset, returns train and test X and y elements
# sampling의 경우 test data는 사용되지 않기에 test_size=0 으로 설정 
def load_dataset(prefix=''):
    # load all train
    total_data = load_dataset_group('train', prefix + 'RES/')
    # trainX, testX = train_test_split(total_data, test_size=0, random_state=123)
    trainX = total_data
    # return trainX, testX
    return trainX


# trainX, testX = load_dataset()
trainX = load_dataset()
# print(trainX.shape)