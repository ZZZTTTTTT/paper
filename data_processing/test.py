from pandas import read_csv
df_kpi = read_csv('./zconfs_bgtg.csv', header=0, index_col=False,encoding='ANSI', dtype = {'kpi_id' : str},names=['deviceCode','point_name','kpi_id','kpi_name'])
df_kpi['point_name']=df_kpi['point_name'].apply(lambda x:str(x).zfill(2))

print(df_kpi)
