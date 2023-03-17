import json
import  pandas as pd
import numpy as np
import os
import datetime,time
#显示所有列
pd.set_option('display.max_columns',None)
root = './data'
file_names = os.listdir(root)
df_list=[]

#print(file_names) 数组
def timestamp_to_time(timestamp):
    # 转换成localtime
    time_local =time.localtime(int(timestamp)/1000)
    # 转换成新的时间格式(精确到秒)
    dt = time.strftime("%Y-%m-%d %H:%M:%S", time_local)
    return dt



for file_name in file_names:
    fileob = root + '/' + file_name
    filename_info=os.path.splitext(file_name)[0].split('_')
    #去除数据项id开头字母
    filename_info[2]=filename_info[2].strip("k")

    #filename_info:['070116M02', '13', 'k18069', 'avg']

    with open(fileob, 'r', encoding='utf-8') as f:
        content = json.load(f)
        json_file_to_df = pd.DataFrame(pd.Series(content), columns=['data_item_value']).reset_index().rename(
            columns={'index': 'timestamp'})
        # print(type(content)) 类型为字典
        # 转化为时间戳
        json_file_to_df['timestamp']=json_file_to_df['timestamp'].apply(timestamp_to_time)
        #插入列
        json_file_to_df.insert(loc=1,column='deviceCode',value=np.repeat(filename_info[0],json_file_to_df.shape[0]))
        json_file_to_df.insert(loc=2,column='point_name',value=np.repeat(filename_info[1],json_file_to_df.shape[0]))
        json_file_to_df.insert(loc=3,column='data_item_id',value=np.repeat(filename_info[2],json_file_to_df.shape[0]))
        json_file_to_df.insert(loc=4,column='polymerization_mode',value=np.repeat(filename_info[3],json_file_to_df.shape[0]))
        # print(json_file_to_df)
        df_list.append(json_file_to_df)
# print(pd.concat(df_list))
# 所有json文件拼接成一个df形状为[63394 rows x 6 columns]
#               timestamp deviceCode point_name data_item_id  \
# 0   2022-10-25 08:00:00  010420M01         01      1241017
# 1   2022-10-26 08:00:00  010420M01         01      1241017
# 2   2022-10-27 08:00:00  010420M01         01      1241017
# 3   2022-10-28 08:00:00  010420M01         01      1241017
# 4   2022-10-29 08:00:00  010420M01         01      1241017