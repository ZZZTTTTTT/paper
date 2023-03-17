#coding:utf-8
import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
from tcn.tcn import TCN
# from darts.models import  TCN
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import layers,Input
import math
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os
bpath = os.getcwd()
import pickle
from keras.models import load_model, Model, Sequential

"""
TCN时空卷积模型
"""
class TCN_demo:
    def __init__(self,
               chunk_size=500000,                   #分块读取文件时，块的大小
               window_size = 7,                     #按窗口创建数据集时,时间窗口的大小
               nb_filters = 20,                     #tcn网络卷积核数
               kernel_size = 46,                    #卷积核大小
               optimizer ="Adam",                   #模型优化器
               loss = "mae",                        #模型损失定义
               epochs =5,                          #训练伦次
               batch_size = 128,                    #批次大小
               validation_split = 0.3,              #验证集比例
               dilations=[int(math.pow(2, i + 1)) for i in range(8)]  # 膨胀大小为2的次方
                 ):
        self.chunk_size = chunk_size
        self.scaler = None
        self.window_size = window_size
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.optimizer = optimizer
        self.loss = loss
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.dilations = dilations
    def load_data(self,path):
        """
        加载数据集
        :param chunk_size:
        :param path:
        :return:
        """
        #分块读取文件
        reader = pd.read_csv(path, header=0, iterator=True, encoding="gb18030")
        chunks = []
        loop = True
        while loop:
            try:
                chunk = reader.get_chunk(self.chunk_size)
                chunks.append(chunk)
            except StopIteration:
                loop = False
        #将块拼接为pandas dataFrame格式
        df = pd.concat(chunks, ignore_index=True)
        return df

    def maxmin_scaler(self,df):
        """
        max min归一化
        :param df:
        :param scaler:
        :return:
        """
        if self.scaler:
            df_s = self.scaler.transform(df.values)
        else:
            self.scaler = MinMaxScaler()
            df_s = self.scaler.fit_transform(df.values)
        return df_s

    def creat_feature(self,df_s,column_len):
        """
        在归一化基础上,根据时间窗口生成特征集和标签集
        :param df_s:
        :param window_size:
        :param column_len:
        :return:
        """
        #按滑窗生成数据集
        X = []
        label = []
        for i in range(len(df_s) - self.window_size):
            X.append(df_s[i:i + self.window_size, :].tolist())
            label.append(df_s[i + self.window_size, :1].tolist()[0])
        #转换数据shape ,转化为（window_size*cols，2）
        X = np.array(X).reshape(-1, self.window_size * column_len, 1)
        label = np.array(label)
        return X, label

    def rmse(self,pred, true):
        """
        根计算预测值和真实值的rmse分数
        :param pred:
        :param true:
        :return:
        """
        return np.sqrt(np.mean(np.square(pred - true)))

    def plot(self,pred, true):
        """
        绘制预测值和真实值曲线
        :param pred:
        :param true:
        :return:
        """
        pred = pred
        true = true
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(range(len(pred)), pred)
        ax.plot(range(len(true)), true)
        plt.xlim(0,200)
        plt.show()

    def do_train(self,path):
        """
        训练并输出模型
        :param path:
        :return:
        """
        ###1.加载数据,标准化,生成数据集
        df_train = self.load_data(path)
        df_s = self.maxmin_scaler(df_train)
        column_len = df_train.shape[1]
        x_train, y_train = self.creat_feature(df_s, column_len)

        ###2.构建网络层级
        inputs = layers.Input(shape=(x_train.shape[1], x_train.shape[2]), name='inputs')

        #神经元（卷积核）20个，卷积核大小46，膨胀大小为2的次方
        t=TCN(return_sequences=False,nb_filters=self.nb_filters,kernel_size=self.kernel_size,dilations=self.dilations)(inputs)
        outputs=layers.Dense(units=1, activation='sigmoid')(t)

        tcn_model=tf.keras.Model(inputs,outputs)
        tcn_model.compile(optimizer=self.optimizer,
                               loss='mae',
                               metrics=['mae'])

        ###3.训练并保存模型
        tcn_model.fit(x_train, y_train, epochs=self.epochs, validation_split=self.validation_split,batch_size = self.batch_size)
        tcn_model.summary()
        tcn_model.save(os.path.join(bpath, 'model.ckpt'))

    def do_predict(self,path):
        """
        预测并输出预测结果
        :param path:
        :return:
        """
        ###1.加载数据,标准化,生成数据集
        df_test = self.load_data( path)
        df_s = self.maxmin_scaler(df_test)
        column_len = df_test.shape[1]
        print("column_len:",column_len)
        x_test, y_test = self.creat_feature(df_s, column_len)

        ###2.加载模型并预测
        tcn_model =  tf.keras.models.load_model(os.path.join(bpath, 'model.ckpt'))
        predict = tcn_model.predict(x_test)

        ###3.预测结果反标准化
        pre_copies = np.repeat(predict, column_len, axis=-1)
        label_copies = np.repeat(y_test, column_len, axis=-1)
        pred = self.scaler.inverse_transform(np.reshape(pre_copies, (len(predict), column_len)))[:, 0].reshape(-1)
        test_label = self.scaler.inverse_transform(np.reshape(label_copies, (len(y_test), column_len)))[:, 0].reshape( -1)

        ###4.评价/绘图
        print('RMSE ', self.rmse(pred, test_label))
        self.plot(pred, test_label)

        ###5.预测结果输出
        df_out = pd.DataFrame()
        df_out["predict"] = pred
        df_out["y"] = test_label
        if not os.path.exists(bpath + r"\result"):
            os.makedirs(bpath + r"\result")
        df_out.to_csv(bpath + r"\result\pre_tcn.csv", index=False)

if __name__ == "__main__":
    demo = TCN_demo()     #加载demo类
    demo.do_train(path=bpath + r"\data\train.csv")  #训练
    demo.do_predict(path=bpath + r"\data\test.csv") #测试