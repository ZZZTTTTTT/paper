from pandas import read_csv
import numpy as np
import scipy.stats
import  pandas as pd


path='./643121M18_06_8k速度波形2_1000.csv'
Fs = 25600  # 采样频率


def load_data(path):

    f=open(path,encoding="utf-8")
    # load dataset
    dataset = read_csv(f)
    dataset.columns=['timestamp','signal']
    dataset['timestamp']=dataset['timestamp'].apply(lambda x: x[:-7])
    # dataset = read_csv('single.csv', header=0, index_col=0,encoding="ANSI")
    #按时间对信号分组
    df_grouped = dataset.groupby('timestamp')['signal'].apply(list).reset_index(name='values')

    return df_grouped
def Time_fea( signal_):
    """
    提取时域特征 11 类
    """
    N = len(signal_)
    y = np.array(signal_)
    t_mean_1 = np.mean(y)                                    # 1_均值（平均幅值）

    t_std_2  = np.std(y, ddof=1)                             # 2_标准差

    t_fgf_3  = ((np.mean(np.sqrt(np.abs(y)))))**2           # 3_方根幅值

    t_rms_4  = np.sqrt((np.mean(y**2)))                      # 4_RMS均方根

    t_pp_5   = 0.5*(np.max(y)-np.min(y))                     # 5_峰峰值  (参考周宏锑师姐 博士毕业论文)

    #t_skew_6   = np.sum((t_mean_1)**3)/((N-1)*(t_std_3)**3)
    t_skew_6   = scipy.stats.skew(y)                         # 6_偏度 skewness

    #t_kur_7   = np.sum((y-t_mean_1)**4)/((N-1)*(t_std_3)**4)
    t_kur_7 = scipy.stats.kurtosis(y)                        # 7_峭度 Kurtosis

    t_cres_8  = np.max(np.abs(y))/t_rms_4                    # 8_峰值因子 Crest Factor

    t_clear_9  = np.max(np.abs(y))/t_fgf_3                   # 9_裕度因子  Clearance Factor

    t_shape_10 = (N * t_rms_4)/(np.sum(np.abs(y)))           # 10_波形因子 Shape fator

    t_imp_11  = ( np.max(np.abs(y)))/(np.mean(np.abs(y)))  # 11_脉冲指数 Impulse Fator

    # t_fea = np.Series({t_mean_1, t_std_2, t_fgf_3, t_rms_4, t_pp_5,
    #                   t_skew_6,   t_kur_7,  t_cres_8,  t_clear_9, t_shape_10, t_imp_11 })

    t_fea = pd.Series([t_mean_1, t_std_2, t_fgf_3, t_rms_4, t_pp_5,
                      t_skew_6,   t_kur_7,  t_cres_8,  t_clear_9, t_shape_10, t_imp_11],index=['时域均值','时域标准差','时域方根幅值','时域RMS均方根','时域峰峰值','时域偏度','时域峭度','时域峰值因子','时域裕度因子','时域波形因子','时域脉冲指数'])
    #print("t_fea:",t_fea.shape,'\n', t_fea)
    return t_fea

def Fre_fea(signal_):
    """
    提取频域特征 13类
    :param signal_:
    :return:
    """
    signal_=np.array(signal_)
    L = len(signal_)
    PL = abs(np.fft.fft(signal_ / L))[: int(L / 2)]
    PL[0] = 0
    f = np.fft.fftfreq(L, 1 / Fs)[: int(L / 2)]
    x = f
    y = PL
    K = len(y)

    #平均值
    f_12 = np.mean(y)
    #方差
    f_13 = np.var(y)
    #偏度
    f_14 = (np.sum((y - f_12)**3))/(K * ((np.sqrt(f_13))**3))
    #峭度
    f_15 = (np.sum((y - f_12)**4))/(K * ((f_13)**2))
    #频率中心
    f_16 = (np.sum(x * y))/(np.sum(y))
    #频率标准差
    f_17 = np.sqrt((np.mean(((x- f_16)**2)*(y))))
    #频率均方根
    f_18 = np.sqrt((np.sum((x**2)*y))/(np.sum(y)))
    #频率方差
    f_19 = np.sqrt((np.sum((x**4)*y))/(np.sum((x**2)*y)))
    #频率峭度
    f_20 = (np.sum((x**2)*y))/(np.sqrt((np.sum(y))*(np.sum((x**4)*y))))
    #形状因子
    f_21 = f_17/f_16
    # 频率偏度
    f_22 = (np.sum(((x - f_16)**3)*y))/(K * (f_17**3))
    # 频率峭度
    f_23 = (np.sum(((x - f_16)**4)*y))/(K * (f_17**4))

    # f_fea = np.array([f_12, f_13, f_14, f_15, f_16, f_17, f_18, f_19, f_20, f_21, f_22, f_23])
    f_fea = pd.Series([f_12, f_13, f_14, f_15, f_16, f_17, f_18, f_19, f_20, f_21, f_22, f_23],
                      index=['频域均值','频域方差','频域偏度','频域峰度','频域频率中心','频域平均绝对偏差','频域平均幅值','频域根幅值平方和','频域波形因子','频域频带宽度','频域斜度','频域峰值因子'])

    #print("f_fea:",f_fea.shape,'\n', f_fea)
    return f_fea



def feature_extract(dataset):
    df_time_feature=dataset['values'].apply(Time_fea)
    df_freq_feature=dataset['values'].apply(Fre_fea)
    return pd.concat([dataset['timestamp'],df_time_feature,df_freq_feature],axis=1)

df_grouped=load_data(path)
feature_extracted=feature_extract(df_grouped)
print(feature_extracted)
feature_extracted.to_csv('./feature.csv', encoding="utf_8_sig", mode="w",index=False)