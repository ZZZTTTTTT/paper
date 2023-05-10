import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller as ADF
#销量数据
filename='./arima_data.xls'
#预测天数
forrecastnum=5
data=pd.read_excel(filename,index_col=u'日期')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
data.plot()
plt.title('Time Series')
plt.show()
#自相关系数
plot_acf(data)
plt.show()
print(u'原始序列的ADF检验结果为：',ADF(data[u'销量']))
#p值为0.9983759421514264，不平稳，所以进行差分，采用的是一阶差分，所以d=1
D_data=data.diff(periods=1).dropna()
D_data.columns=[u'销量差分']
D_data.plot()
plt.show()
#自相关系数和偏自相关系数
plot_acf(D_data).show()
plot_pacf(D_data).show()
print(u'1阶差分序列的ADF检验结果为：',ADF(D_data[u'销量差分']))
#p值为0.02267343544004886，一阶差分结果平稳
from statsmodels.stats.diagnostic import acorr_ljungbox
print(u'差分序列的白噪声检验结果为：',acorr_ljungbox(D_data,lags=1))
#p值为0.000773，白噪声检验结果合格（如果是白噪声，即纯随机序列，则没有研究的意义了。）
#from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
data[u'销量'] = data[u'销量'].astype(float)
pmax=int(len(D_data)/10)
qmax=int(len(D_data)/10)
bic_matrix=[]
for p in range(pmax+1):
    tmp=[]
    for q in range(qmax+1):
            #print(ARIMA(data,order=(p,1,q)).fit().bic)
            try:
                tmp.append(sm.tsa.arima.ARIMA(data, order=(p, 1, q)).fit().bic)
            except:
                tmp.append(None)
    bic_matrix.append(tmp)
bic_matrix=pd.DataFrame(bic_matrix)
#模型评估标准：BIC值越低越好。
print(u'bic_matrix结果为：',bic_matrix)
p,q=bic_matrix.stack().idxmin()
print(u'bic最小的P值和q值为：%s、%s'%(p,q))
#模型拟合
model=sm.tsa.arima.ARIMA(data, order=(p, 1, q)).fit()
model.summary()
print(model.summary())
#模型预测
forecast=model.forecast(5)
print(forecast)





