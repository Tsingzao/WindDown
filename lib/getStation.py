import pandas as pd
from lib.config import *


def getLabel():
    label = pd.read_csv(tsvPath, sep='\t')
    # label = label[label['站号']=='F3']
    label['time'] = label['日期']+' '+label['时间']
    label = label.drop('日期', axis=1)
    label = label.drop('时间', axis=1)
    label['time'] = pd.to_datetime(label['time'])
    label = label.rename(columns={'站号':'id', '温度(℃)':'t', '湿度(%)':'h', '风速(m/s)':'s', '风向(°)':'d', '气压(hPa)':'p'})
    return label