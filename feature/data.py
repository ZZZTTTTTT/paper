from pandas import read_csv
from pandas import DataFrame


def load_data():
    # load dataset
    dataset = read_csv('./643121M18_06.csv', header=0, index_col=0, encoding="ANSI")
    # dataset = read_csv('single.csv', header=0, index_col=0,encoding="ANSI")

    return dataset

data=load_data()
print(data)