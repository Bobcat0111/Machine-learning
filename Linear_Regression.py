'''
Created on 2019年11月19日

@author: genzhou
'''
from __future__ import print_function
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import numpy as np
import csv
import pandas as pd
import time
# set the random seeds to make sure your results are reproducible
from numpy.random import seed
seed(1)
from tensorflow.compat.v1 import set_random_seed
set_random_seed(1)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# specify path to training data and testing 
# reading train data
time_start = time.time()

print("Reading training data")

ytr=pd.read_csv("train_selection_0.1_dist.csv")
test=pd.read_csv("test_clean_with_distance.csv")
#ytr=pd.DataFrame(df[0:1000001])
pd.set_option('display.max_column',15)
pd.set_option('display.width',200)

print(ytr.corr())
#remove fare<0
print("Old Size: {oldsize}".format(oldsize=len(ytr)))
df_train=ytr.drop(ytr[ytr['fare_amount']<0].index,axis=0)
print("New Size: {newsize}".format(newsize=len(df_train)))
print(df_train.corr())
#remove passenger >6
print("Old Size: {oldsize}".format(oldsize=len(df_train)))
df_train1=df_train.drop(df_train[df_train['passenger_count']>6].index,axis=0)
print("New Size: {newsize}".format(newsize=len(df_train1)))
print(df_train1.corr())
#remove bad longitude and latitude
print("Old Size: {oldsize}".format(oldsize=len(df_train1)))
df_train2 = df_train1.drop(((df_train1[df_train1['pickup_latitude']<-90])|(df_train1[df_train1['pickup_latitude']>90])|(df_train1[df_train1['pickup_latitude']==0]))
         .index,axis=0)
df_train3 = df_train2.drop(((df_train2[df_train2['pickup_longitude']<-180])|(df_train2[df_train2['pickup_longitude']>180])|(df_train2[df_train2['pickup_longitude']==0]))
         .index,axis=0)
df_train4 = df_train3.drop(((df_train3[df_train3['dropoff_latitude']<-90])|(df_train3[df_train3['dropoff_latitude']>90])|(df_train3[df_train3['pickup_latitude']==0]))
         .index,axis=0)
df_train5 = df_train4.drop(((df_train4[df_train4['dropoff_longitude']<-180])|(df_train4[df_train4['dropoff_longitude']>180])|(df_train4[df_train4['pickup_longitude']==0]))
         .index,axis=0)
print("New Size: {newsize}".format(newsize=len(df_train5)))
print(df_train5.corr())
#remove  fare<2.5
print("Old Size: {oldsize}".format(oldsize=len(df_train5)))
df_train6 = df_train5.drop(df_train5.loc[(df_train5['diff']==0) & (df_train5['fare_amount'] < 2.5)].index,axis=0)
print("New Size: {newsize}".format(newsize=len(df_train6)))
print(df_train6.corr())
#remove distance>50000
print("Old Size: {oldsize}".format(oldsize=len(df_train6)))
df_train7 = df_train6.drop(df_train6.loc[(df_train6['diff']>100000)&(df_train6['fare_amount']!=0)].index,axis=0)
print("New Size: {newsize}".format(newsize=len(df_train7)))
print(df_train7.corr())
'''
ytr=pd.read_csv("train_with_distance.csv.csv")
df=pd.DataFrame(ytr[0:1000001])
    
    #new_df=pd.DataFrame.from_dict(document, orient='index')
df.to_csv('train_clean10000000.csv')
'''
#print(ytr[0:1])
x=df_train7[['pickup_datetime','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','passenger_count','diff']]
y=df_train7.fare_amount
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
model=LinearRegression()
model.fit(x_train,y_train)
print('\n')
print('Coefficient of Determination: ',model.score(x_test,y_test),'\n')
print('Regression Coefficient: ',model.coef_,'\n')
print('Intercept: ',model.intercept_,'\n')
predict=model.predict(test)
submission = pd.read_csv('sample_submission.csv')
submission['fare_amount'] = predict
submission.to_csv('submission_1.csv', index=False)

time_end = time.time()
print("Time of Running: ", time_end - time_start,'\n')
