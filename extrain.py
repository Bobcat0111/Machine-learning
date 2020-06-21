'''
Created on 2019年11月20日

@author: genzhou
'''
from __future__ import print_function
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import numpy as np
import csv
import pandas as pd
import math
from math import radians, cos, sin, asin, sqrt
# set the random seeds to make sure your results are reproducible
from numpy.random import seed
seed(1)
from tensorflow.compat.v1 import set_random_seed
set_random_seed(1)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
'''
# specify path to training data and testing 
#读取学习数据
print("Reading training data")
'''
xtr=pd.read_csv("test_clean.csv")
'''
xtr=pd.DataFrame(ytr[0:1000001])

    #new_df=pd.DataFrame.from_dict(document, orient='index')
df.to_csv('train_clean10000000.csv')

pandas_new=pd.read_csv("train_clean.csv")
pd.set_option('display.max_column',15)
pd.set_option('display.width',200)
print(pandas_new.head())
'''
'''
xtr=pd.read_csv("train_selection_0.1.csv")
'''
on_lo=xtr.pickup_longitude
on_la=xtr.pickup_latitude
off_lo=xtr.dropoff_longitude
off_la=xtr.dropoff_latitude
difflo=[0 for i in range(len(xtr))]
diffla=[0 for i in range(len(xtr))]
diff=[0 for i in range(len(xtr))]

def distance(lon1, lat1, lon2, lat2): # 经度1，纬度1，经度2，纬度2 （十进制度数）
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # 将十进制度数转化为弧度
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
 
    # haversine公式
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # 地球平均半径，单位为公里
    return c * r * 1000

for i in range(len(xtr)):
    diff[i]=distance(on_lo[i],on_la[i],off_lo[i],off_la[i])
    


dataframe=pd.DataFrame({'pickup_datetime':xtr.pickup_datetime,'pickup_longitude':xtr.pickup_longitude,'pickup_latitude':xtr.pickup_latitude,'dropoff_longitude':xtr.dropoff_longitude,'dropoff_latitude':xtr.dropoff_latitude,'passenger_count':xtr.passenger_count,'distance':diff})
dataframe.to_csv("test_clean_with_distance.csv",index=False)
data=pd.read_csv("test_clean_with_distance.csv")
print(data.head())

