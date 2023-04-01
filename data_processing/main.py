import json
import  pandas as pd
import numpy as np
import os
import datetime,time
from pandas import read_csv
from functools import reduce
#显示所有列
pd.set_option('display.max_columns',None)


df_kpi = read_csv('./zconfs_bgtg.csv', index_col=False,encoding='ANSI', dtype = {'kpi_id' : str},names=['deviceCode','point_name','kpi_id','kpi_name'])
df_kpi['point_name']=df_kpi['point_name'].apply(lambda x:str(x).zfill(2))


# print(df_kpi)
#[15661 rows x 4 columns]
#       deviceCode  point_name     kpi_id  kpi_name
# 0      150208M02           02  1241013.0   高频加速度峭度
# 1      150208M02           04  1241013.0   高频加速度峭度
# 2      150208M02           03  1241013.0   高频加速度峭度
# 3      150402M01           09  1241012.0  高频加速度RMS


"""
转化为时间戳格式
"""
def timestamp_to_time(timestamp):
    # 转换成localtime
    time_local =time.localtime(int(timestamp)/1000)
    # 转换成新的时间格式(精确到秒)
    dt = time.strftime("%Y-%m-%d %H:%M:%S", time_local)
    return dt
#特殊处理设备070711M01
def timestamp_to_time2(timestamp):
    # 转换成localtime
    time_local =time.localtime((int(timestamp)/1000)+100)
    # 转换成新的时间格式(精确到秒)
    dt = time.strftime("%Y-%m-%d %H:%M:%S", time_local)
    return dt
"""
查kpi_name
"""
def get_kpi_name(filename_info):

    content=df_kpi[(df_kpi['deviceCode']==filename_info[0]) & (df_kpi['point_name']==filename_info[1]) & (df_kpi['kpi_id']==filename_info[2])]

    if(len(content['kpi_name'])):
        return content['kpi_name'].values[0]
    else:
        print('没有该kpi_name')
def merge_df_list(df_list,key):
    df_merge = reduce(lambda left, right: pd.merge(left, right, on=[key], how='left'), df_list)
    return df_merge
def get_devices_df(path,device_code,point_name,method='avg'):
    """

    :param path: 文件路径
    :param device_code: 设备编码
    :param point_name: 测点编号
    :return: timestamp  高频加速度有效值 ...

    """
    file_names = os.listdir(path)
    df_list = []
    for file_name in file_names:
        fileob = path + '/' + file_name
        filename_info=os.path.splitext(file_name)[0].split('_')
        #去除数据项id开头字母
        filename_info[2]=filename_info[2].strip("k")

        #filename_info:['070116M02', '13', 'k18069', 'avg']


        if((filename_info[0]==device_code) & (filename_info[1]==point_name) ):
            kpi_name = get_kpi_name(filename_info)
            content={}
            with open(fileob, 'r', encoding='utf-8')as f:
                json_objects=[json.loads(line) for line in f]
            for obj in json_objects:
                try:
                    content.update(obj)
                except:
                    print('error',obj)

            json_file_to_df = pd.DataFrame(pd.Series(content), columns=[locals()['kpi_name']]).reset_index().rename(
                columns={'index': 'timestamp'})
            # 转化为时间戳
            if(device_code=='070711M01'):
                json_file_to_df['timestamp']=json_file_to_df['timestamp'].apply(timestamp_to_time2)
            else:
                json_file_to_df['timestamp']=json_file_to_df['timestamp'].apply(timestamp_to_time)

            df_list.append(json_file_to_df)
    if len(df_list):
        return merge_df_list(df_list,'timestamp')
    else:
        print('没有kpi数据')

if __name__ == '__main__':
    root = './data'
    #070704M01，070711M01，070712M01
    device1=get_devices_df(root,'070704M01','01')
    device2=get_devices_df(root,'070711M01','01')
    device3=get_devices_df(root,'070712M01','01')
    devices_df=merge_df_list([device1,device2,device3],'timestamp')
    # print(device2)

    # device2.to_csv('./device2.csv', encoding="utf_8_sig", mode="w",index=False)
    devices_df.to_csv('./result.csv', encoding="utf_8_sig", mode="w",index=False)
