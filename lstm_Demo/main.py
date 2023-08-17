from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import numpy as np
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from keras.optimizers import Adam
from keras.initializers import Constant
from sklearn.preprocessing import LabelEncoder
from keras.regularizers import l2
import pandas as pd
import abnormal_detect

from matplotlib.pyplot import MultipleLocator
from keras.layers import RepeatVector
from keras.layers import TimeDistributed






pd.set_option('display.max_columns',1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth',1000)



class LSTM_Demo:
    def __init__(self,n_hours = 3):
        self.n_hours=n_hours
        self.n_features=12
        self.model=None
        self.scaler=None
        self.history=None
        self.n_seq=3
        self.n_batch=50
    def load_data(self):
        # load dataset
        dataset = read_csv('../data_processing/result.csv', header=0, index_col=0,encoding="ANSI")
        # dataset = read_csv('single.csv', header=0, index_col=0,encoding="ANSI")
        self.n_features=dataset.shape[1]
        return dataset

    def split_train_test(self,reframed):

        # split into train and test sets
        values = reframed.values

        n_train_hours = 13460
        train = values[:n_train_hours, :]
        test = values[n_train_hours:, :]

        # split into input and outputs
        n_obs = self.n_seq * self.n_features
        # 有60=(12*(3+2))列数据
        train_X, train_y = train[:, :-n_obs], train[:,-n_obs :]
        test_X, test_y = test[:, :-n_obs], test[:, -n_obs:]
        return train_X,test_X, test_y, train_y
    def maxmin_scaler(self,dataset):

        #处理缺失值
        dataset=dataset.fillna(dataset.mean())
        # print(dataset.isnull().sum())
        values = dataset.values
        values = values.astype('float32')
        # 标准化/放缩 特征值在（0,1）之间
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = self.scaler.fit_transform(values)
        return scaled

    def series_to_supervised(self,data, n_in=1, n_out=1, dropnan=True):

        n_vars = 1 if type(data) is list else data.shape[1]
        df = DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        # 将3组输入数据依次向下移动3，2，1行，将数据加入cols列表（技巧：(n_in, 0, -1)中的-1指倒序循环，步长为1）
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        # 将一组输出数据加入cols列表（技巧：其中i=0）
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
        # cols列表(list)中现在有四块经过下移后的数据(即：df(-3),df(-2),df(-1),df)，将四块数据按列 并排合并
        agg = concat(cols, axis=1)
        # 给合并后的数据添加列名
        agg.columns = names
        # 删除NaN值列
        if dropnan:
            agg.dropna(inplace=True)
        print(agg)
        return agg
    def get_splited_data(self):
        dataset=self.load_data()
        #平稳性检验p值8.429351731055096e-11，是平稳的

        #一阶差分
        #归一化
        scaled=self.maxmin_scaler(dataset)


        # 转换成监督数据，四列数据，3->1，三组预测一组
        # 用3小时数据预测一小时数据，10个特征值

        # 构造一个3->1的监督学习型数据
        reframed = self.series_to_supervised(scaled, self.n_hours, self.n_seq)
        train_X,test_X, test_y, train_y=self.split_train_test(reframed)
        # 将数据转换为3D输入，timesteps=3，3条数据预测1条 [samples, timesteps, features]
        train_X = train_X.reshape((train_X.shape[0], self.n_hours, self.n_features))
        test_X = test_X.reshape((test_X.shape[0], self.n_hours, self.n_features))
        print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
        return train_X,test_X, test_y, train_y

    def do_train(self):
        train_X, test_X, test_y, train_y = self.get_splited_data()


        # 设计网络
        self.model = Sequential()
        # kernel_initializer = Constant(value=1.0), bias_initializer = 'zeros'
        self.model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2]), kernel_initializer=Constant(value=1.0), bias_initializer='zeros'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(train_y.shape[1]))
        self.model.compile(loss='mae', optimizer=Adam(learning_rate=1e-3))
        # 拟合网络
        self.history = self.model.fit(train_X, train_y, epochs=125, batch_size=self.n_batch, validation_data=(test_X, test_y), verbose=1,
                            shuffle=False)

        #多步预测
        self.mutil_step_predict(test_X,test_y)


    def do_ED_train(self):
        train_X, test_X, test_y, train_y = self.get_splited_data()
        # reshape output into [samples, timesteps, features]
        train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
        test_y = test_y.reshape((test_y.shape[0], test_y.shape[1], 1))

        # 设计网络
        self.model = Sequential()
        self.model.add(
            LSTM(120, activation='relu', input_shape=(train_X.shape[1], train_X.shape[2]),
                 kernel_initializer='glorot_uniform'))
        # self.model.add(Dropout(0.2))
        self.model.add(RepeatVector(train_y.shape[1]))
        self.model.add(LSTM(120, activation='relu', return_sequences=True))
        self.model.add(TimeDistributed(Dense(40, activation='relu')))
        self.model.add(TimeDistributed(Dense(1)))
        self.model.compile(loss='mae', optimizer=Adam(learning_rate=0.001))
        # fit network
        self.history = self.model.fit(train_X, train_y, epochs=50, batch_size=100, verbose=1,
                                      validation_data=(test_X, test_y), shuffle=False)
        print(self.history.history)
        print(self.history.history['loss'])
        print(self.history.history['val_loss'])

        #多步预测
        self.mutil_step_predict(test_X,test_y)

    def do_predict(self):
        train_X, test_X, test_y, train_y = self.get_splited_data()

        # 执行预测
        model=self.model
        yhat = model.predict(test_X)
        # 将数据格式化成 n行 * 24列
        test_X = test_X.reshape((test_X.shape[0], self.n_hours * self.n_features))
        # 将预测列据和后7列数据拼接，因后续逆缩放时，数据形状要符合 n行*8列 的要求
        inv_yhat = concatenate((yhat, test_X[:, (-self.n_features+1):]), axis=1)
        # 对拼接好的数据进行逆缩放
        inv_yhat = self.scaler.inverse_transform(inv_yhat)
        inv_yhat = inv_yhat[:, 0]

        test_y = test_y.reshape((len(test_y), 1))
        # 将真实列据和后7列数据拼接，因后续逆缩放时，数据形状要符合 n行*8列 的要求
        inv_y = concatenate((test_y, test_X[:, (-self.n_features+1):]), axis=1)
        # 对拼接好的数据进行逆缩放
        inv_y = self.scaler.inverse_transform(inv_y)
        inv_y = inv_y[:, 0]

        # 计算RMSE误差值
        rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
        self.plot(inv_y,inv_yhat)
        print(inv_y)
        print(inv_yhat)
        print('Test RMSE: %.3f' % rmse)

    # LSTM 单步预测
    def forecast_lstm(self,X):
        X = X.reshape(1,X.shape[0],X.shape[1])
        # 预测张量形状
        forecast = self.model.predict(X, batch_size=self.n_batch)
        # 将预测结果[[XX,XX,XX]]转换成list数组
        return [x for x in forecast[0, :]]

    # 用模型进行预测
    def make_forecasts(self,test):
        forecasts = list()
        # 对X值进行逐个预测
        for i in range(len(test)):

            X = test[i, :]
            # LSTM 单步预测
            forecast = self.forecast_lstm( X )
            # 存储预测数据
            forecasts.append(forecast)
        return forecasts

    #多步预测
    def mutil_step_predict(self,test_X,test_y):
        forecasts = self.make_forecasts(test_X)
        # 将预测后的数据逆转换
        forecasts =self.inverse_transform(forecasts)
        # 从测试数据中分离出y对应的真实值
        actual = [row[:] for row in test_y]
        # 对真实值逆转换
        actual=self.inverse_transform(actual)
        # 评估预测值和真实值的RSM
        self.evaluate_forecasts(actual, forecasts, self.n_seq)
        #作图
        self.plot_forecasts(actual,forecasts,len(test_y)+self.n_seq)
    def inverse_transform(self,forecasts):
        inverted = list()
        for i in range(len(forecasts)):
            # create array from forecast
            forecast = np.array(forecasts[i])
            forecast = forecast.reshape(self.n_seq,self.n_features)

            # 对拼接好的数据进行逆缩放
            inv_yhat = self.scaler.inverse_transform(forecast)
            # inv_yhat = inv_yhat[ 0,:]

            inverted.append(inv_yhat)
        return inverted

    # 评估预测结果的均方差
    def evaluate_forecasts(self,test, forecasts, n_seq):
        for i in range(n_seq):
            actual = [row[i][-4] for row in test]
            predicted = [forecast[i][-4] for forecast in forecasts]
            rmse = sqrt(mean_squared_error(actual, predicted))
            print('t+%d RMSE: %f' % ((i + 1), rmse))

    def plot(self,inv_y,inv_yhat):
        # plot history
        pyplot.subplot(211)
        pyplot.plot(self.history.history['loss'], label='loss')
        pyplot.plot(self.history.history['val_loss'], label='val_loss')
        pyplot.legend()

        # pyplot.plot(self.history.history['acc'], label='acc')
        # pyplot.plot(self.history.history['val_acc'], label='val_acc')
        pyplot.subplot(223)
        pyplot.plot(inv_y, label='y')
        pyplot.plot(inv_yhat, label='yhat')
        pyplot.legend()
        pyplot.show()

    # 多步预测作图
    def plot_forecasts(self,series, forecasts, n_test):
        pyplot.subplot(211)

        pyplot.plot(self.history.history['loss'], label='loss')
        pyplot.plot(self.history.history['val_loss'], label='val_loss')
        pyplot.legend()
        ax = pyplot.gca()
        y_major_locator = MultipleLocator(0.025)
        # ax为两条坐标轴的实例
        ax.yaxis.set_major_locator(y_major_locator)

        pyplot.subplot(223)
        actual=[row[0][-4] for row in series]
        # plot the entire dataset in blue
        pyplot.plot(actual)
        # pyplot.plot([row[0] for row in forecasts])
        # plot the forecasts in red
        for i in range(len(forecasts)):
            xaxis = [x for x in range(i, i+self.n_seq)]
            yaxis = [ forecasts[i][x][-4] for x in range(0, self.n_seq)]
            pyplot.plot(xaxis, yaxis, color='g',linewidth=1,linestyle='--')
        #圈出异常点(这里是按第一个找异常点)
        abnormal_detect.detect_outline(actual, [row[0][-4] for row in forecasts])
        # show the plot
        pyplot.show()


if __name__ == "__main__":
    demo = LSTM_Demo()     #加载类
    # demo.do_train()
    demo.do_ED_train()
    # demo.do_TCN_train()
    # demo.do_predict()
