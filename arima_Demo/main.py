import json
import  pandas as pd
##处理成时间序列数据 字段
f = open('./data.txt','r',encoding="utf8")
json_data=json.load(f)
data_df = pd.DataFrame(json_data)
data_df=data_df[['arisingTime','value']]
data_df.to_excel("./时间序列.xlsx",encoding="utf-8")

