import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from IPython.display import display
import datetime
import time
import math
import warnings
from numpy import asarray
from numpy import fft
from numpy import abs
from numpy import angle
from numpy import copy
from statsmodels.tsa.arima_model import ARIMA
import time
warnings.filterwarnings("ignore")
import glob
from plotting import plot_df, plot_energy
################################
def read_label():
    label = {}
    for i in range(1, 7):
        hi = 'D:/Datasets/REDD/low_freq/house_{}/labels.dat'.format(i)
        label[i] = {}
        with open(hi) as f:
            for line in f:
                splitted_line = line.split(' ')
                label[i][int(splitted_line[0])] = splitted_line[1].strip() + '_' + splitted_line[0]
    return label
################################
def read_merge_data(house, labels):
    path = 'D:/Datasets/REDD/low_freq/house_{}/'.format(house)
    file = path + 'channel_1.dat'
    df = pd.read_table(file, sep = ' ', names = ['unix_time', labels[house][1]], 
                                       dtype = {'unix_time': 'int64', labels[house][1]:'float64'}) 
    
    num_apps = len(glob.glob(path + 'channel*'))
    for i in range(2, num_apps + 1):
        file = path + 'channel_{}.dat'.format(i)
        data = pd.read_table(file, sep = ' ', names = ['unix_time', labels[house][i]], 
                                       dtype = {'unix_time': 'int64', labels[house][i]:'float64'})
        df = pd.merge(df, data, how = 'inner', on = 'unix_time')
    df['timestamp'] = df['unix_time'].astype("datetime64[s]")
    df = df.set_index(df['timestamp'].values)
    df.drop(['unix_time','timestamp'], axis=1, inplace=True)
    return df


###############Train Test Data on house 1
# Separate house 1 data into train, validation and test data
def data_separation(df, dates):
    df1_train = df[1][:dates[1][17]]
    df1_val = df[1][dates[1][17]:dates[1][20]]
    df1_test = df[1][dates[1][20]:]
    print('df_train.shape: ', df1_train.shape)
    print('df_val.shape: ', df1_val.shape)
    print('df_test.shape: ', df1_test.shape)
    print(dates[1])
    # Using mains_1, mains_2 to predict refrigerator
    X_train1 = df1_train[['mains_1','mains_2']].values
    y_train1 = df1_train['refrigerator_5'].values
    X_val1 = df1_val[['mains_1','mains_2']].values
    y_val1 = df1_val['refrigerator_5'].values
    X_test1 = df1_test[['mains_1','mains_2']].values
    y_test1 = df1_test['refrigerator_5'].values
    print(X_train1.shape, y_train1.shape, X_val1.shape, y_val1.shape, X_test1.shape, y_test1.shape)
    return X_train1, y_train1, X_val1, y_val1, X_test1, y_test1, df1_test
def process_data(df, dates, x_features, y_features, look_back = 50):
    i = 0
    for date in dates:
        data = df[date]
        data_shift = df[date].shift(periods=1, fill_value=0)
        len_data = data.shape[0]
        x = np.array([data[x_features].values[i:i+look_back] 
                      for i in range(len_data - look_back) ]).reshape(-1,look_back, np.shape(x_features)[0])
        time.sleep(10)
        z = np.array([data_shift[y_features].values[i:i+look_back] 
                    for i in range(len_data - look_back) ]).reshape(-1,look_back, np.shape(y_features)[0])
        y = data[y_features].values[look_back:,:]
        if i == 0:
            X = x
            Z = z
            Y = y
        else:
            X = np.append(X, x, axis=0)
            Z = np.append(Z, z, axis=0)
            Y = np.append(Y, y, axis=0)
        i += 1
    return X, Y, Z

def get_fft(values):

    result = pd.DataFrame()
    close_fft = fft.fft(values)
    fft_df = DataFrame({'fft': close_fft})
    fft_df['absolute'] = fft_df['fft'].apply(lambda x: abs(x))
    fft_df['angle'] = fft_df['fft'].apply(lambda x: angle(x))
    fft_list = asarray(fft_df['fft'].tolist())
    plt.figure(figsize=(14, 7), dpi=100)
    for num_ in [3, 20, 100]:
        fft_list_m10 = copy(fft_list)
        fft_list_m10[num_:-num_] = 0
        result["fft_{}".format(num_)] = fft.ifft(fft_list_m10).real
        plt.plot(fft.ifft(fft_list_m10),label='Fourier transform with {} components'.format(num_))
    plt.plot(values[9000:1400],  label='Real')
    plt.xlabel('Time')
    plt.ylabel('Power')
    plt.title('FFT on REDD dataset')
    plt.legend()
    plt.savefig(fname='FFT on REDD dataset')
    plt.close()
    return result

def get_arima(values):
    X = values
    size = int(len(X) * 0.9)
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=(5, 1, 0))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
    return history

def get_technical_indicators(dataset):

    dfIndex = pd.DataFrame()

    # Create 7 and 21 days Moving Average
    dfIndex['ma7'] = dataset[1]['mains_1'].rolling(window=7).mean()
    dfIndex['ma21'] = dataset[1]['mains_1'].rolling(window=21).mean()

    # Create MACD
    dfIndex['26ema'] = dataset[1]['mains_1'].ewm(span=26).mean()
    dfIndex['12ema'] = dataset[1]['mains_1'].ewm(span=12).mean()
    dfIndex['MACD'] = (dfIndex['12ema']-dfIndex['26ema'])

    # Create Bollinger Bands
    dfIndex['20sd'] = dataset[1]['mains_1'].rolling(20).std()
    dfIndex['upper_band'] = dfIndex['ma21'] + (dfIndex['20sd']*2)
    dfIndex['lower_band'] = dfIndex['ma21'] - (dfIndex['20sd']*2)

    # Create Exponential moving average
    dfIndex['ema'] = dataset[1]['mains_1'].ewm(com=0.5).mean()

    
    # Create Momentum
    dfIndex['momentum'] = dataset[1]['mains_1']-1

    # Dropping
    dfIndex.drop(['12ema', 'ma7', 'momentum', 'upper_band', 'lower_band'], axis=1, inplace=True)
    # Create FFT
    dfFFT = get_fft(dataset[1]['mains_1'].values)
    dfFFT.set_index(dataset[1].index, inplace=True)
    # Create ARIMA
    #dfARIMA = get_arima(dataset[1]['mains_1'].values)

    # Concatenation 
    frames = [dfIndex, dfFFT, dataset[1]]
    result = pd.concat(frames, axis=1)
    result.fillna(0, inplace=True)
    return result




def prepare_data():
    labels = read_label()
    for i in range(1,3):
        print('House {}: '.format(i), labels[i], '\n')
    df = {}
    dates = {}
    for i in range(1,3):
        df[i] = read_merge_data(i, labels)
        print('House {} data has shape: '.format(i), df[i].shape)
        display(df[i].tail(3))
        dates[i] = [str(time)[:10] for time in df[i].index.values]
        dates[i] = sorted(list(set(dates[i])))
        print('House {0} data contain {1} days from {2} to {3}.'.format(i,len(dates[i]),dates[i][0], dates[i][-1]))
        print(dates[i], '\n')
    # Plotting
    for i in range(1,3):
        plot_df(df[i][:dates[i][1]], 'First 2 day data of house {}'.format(i))
    plot_energy(df)
    df[1] = get_technical_indicators(df)
    start = time.time()
    X_train, y_train, z_train = process_data(df[1], dates[1][:20], df[1].columns.values[:10], df[1].columns.values[10:])
    X_test, y_test, _ = process_data(df[1], dates[1][20:], df[1].columns.values[:10], df[1].columns.values[10:])
    X_train1, y_train1, X_val1, y_val1, X_test1, y_test1, df1_test = data_separation(df, dates)
    print('Process data time: ', time.time() - start)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    return X_train, y_train, z_train, X_test, y_test, df1_test, dates
