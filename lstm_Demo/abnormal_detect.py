
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


x_values = []
y_values = []


# 在一维数据集上检测离群点的函数
def find_anomalies(random_data):
    # 将上、下限设为3倍标准差
    random_data_std = np.std(random_data)
    random_data_mean = np.mean(random_data)
    anomaly_cut_off = random_data_std * 3

    lower_limit = random_data_mean - anomaly_cut_off
    upper_limit = random_data_mean + anomaly_cut_off
    print("下限： ", lower_limit)
    print("上限： ", upper_limit)
    # 异常
    for i in range(len(random_data)):

        if random_data[i] > upper_limit or random_data[i] < lower_limit:
            x_values.append(i)
            # y_values.append(random_data[i])

    return x_values


# r为检测模型的阈值参数
def detect_outline(y, y_hat):
    residual = np.array(y)  - np.array(y_hat)
    x_values = find_anomalies(residual)
    y_values=[y[i] for i in x_values]
    plot_outline(x_values, y_values)



def plot_outline(x_values, y_values):
    plt.scatter(x_values, y_values, s=10,c='none',edgecolors='r')