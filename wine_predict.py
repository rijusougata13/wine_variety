#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 22:43:33 2020

@author: sougata
"""

#===============================================================>
#data preprocessing before training
#===============================================================>


import pandas as pd
import numpy as np

train_data=pd.read_csv('train.csv')
test_data=pd.read_csv('test.csv')
data=train_data.append(test_data,ignore_index=True)
train_index=len(train_data)
test_index=len(data)-len(test_data)


data['country']=data['country'].fillna(data['country'].mode()[0])
data['province']=data['province'].fillna(data['province'].mode()[0])
data['price']=data['price'].fillna(data['price'].mean())

#handling description
for i in range(len(data.index)):
    data.loc[i, 'review_description'] = len(data.loc[i, 'review_description'])


#handling desgnation
null_ID = data[pd.isnull(data['designation'])].index
notnull_ID = list(set(data.index) - set(null_ID))
des = data.iloc[notnull_ID]
des = des['designation'].unique()
des_dict = dict()
for i in range(len(des)):
    des_dict[des[i]] = i
# print(des_dict)
for d in des:
    data['designation'] = data['designation'].replace(d, des_dict[d])
data.loc[null_ID, ['designation']] = -1

#handling point
point_mean = data['points'].mean()
data.loc[(data['points'] > 96) & (data['points'] <= 100), 'points'] = 1
data.loc[(data['points'] > 92) & (data['points'] <= 96), 'points'] = 2
data.loc[(data['points'] > 88) & (data['points'] <= 92), 'points'] = 3
data.loc[(data['points'] > 84) & (data['points'] <= 88), 'points'] = 4
data.loc[(data['points'] >= 80) & (data['points'] <= 84), 'points'] = 5
data.head(10)

#handling region_1 ,region_2 and province
data.loc[pd.isnull(data['region_2']),['region_2']]=''
data.loc[pd.isnull(data['region_1']), ['region_1']] = ''
data.loc[:,'region']=data.loc[:,'province']+' '+data.loc[:,'region_1']+' '+data.loc[:,'region_2']
del data['region_1']
del data['region_2']
del data['province']
del data['user_name']
del data['review_title']

#encoding region and winery and country
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['region']=le.fit_transform(data['region'])
data['winery']=le.fit_transform(data['winery'])
data['country']=le.fit_transform(data['country'])
#price
price = data['price'].unique()
for i in price:
    data['price'] = data['price'].replace(i, np.floor(i / 100) + 1)


#encoding variety
variety=data.variety.unique()
variety_dict=dict()
for i in  range(len(variety)):
    variety_dict[variety[i]]=i
for var in variety:
    data.variety=data.variety.replace(var,variety_dict[var])


#convert dataset to csv
data.to_csv('processed_data.csv')




#===============================================================>
#training part
#===============================================================>


import pandas as pd
import numpy as np

data=pd.read_csv('processed_data.csv')

test_index=82657
train_index=82657

train=data[:train_index]
train.dropna(inplace=True)
test=data[test_index:]

x_train=train.drop(['variety'],axis=1)
y_train=train['variety']

test=test.drop(['variety'],axis=1)


metrics=list(x_train.columns)
metrics.remove('points')

def standardize(raw_data):
    return ((raw_data - np.mean(raw_data, axis=0)) / np.std(raw_data, axis=0))
x_train[metrics] = standardize(x_train[metrics])
test[metrics] = standardize(test[metrics])

del data['Unnamed: 0']
#===============================================================>
#training model
#===============================================================>

from sklearn.model_selection import train_test_split
x_train1, x_test1, y_train1, y_test1 = train_test_split(x_train, y_train, test_size=0.2)


#===============================================================>
#try to load model otherwise train the model
#===============================================================>

try:
    import pickle
    with open('wine_model.pkl', 'rb') as f:
        classifier = pickle.load(f)

    
except:
    import xgboost
    
    from sklearn.ensemble import RandomForestRegressor 
    classifier=xgboost.XGBClassifier(n_estimators=100,
                                   max_depth=30,
                                   random_state=0)
    classifier.fit(x_train1,y_train1)



#===============================================================>
#predict the value in test.csv
#===============================================================>


prediction=classifier.predict(test)
predicted_value=['variety']
for i in prediction:
    predicted_value.append(variety[int(i)])



#===============================================================>
#calculating the accuracy in train.csv dividing them into train and testing data
#===============================================================>

from sklearn.metrics import accuracy_score
y_test1_predict=classifier.predict(x_test1)
print(' on train.csv accuracy-> {}'.format(accuracy_score(y_test1,y_test1_predict)))

#===============================================================>
#saving and restoring a model
#===============================================================>

import pickle
with open('wine_model.pkl', 'wb') as f:
    pickle.dump(classifier, f)
'''
with open('wine_model.pkl', 'rb') as f:
    classifier = pickle.load(f)
'''

#===============================================================>
#converting to a csv file 
#===============================================================>
testing_data=pd.read_csv('test.csv')
output=pd.DataFrame({'user_name':test_data.user_name,'country':test_data.country,'review_title':test_data.review_title,'review_description':test_data.review_description,'designation':test_data.designation,'points':test_data.points,'province':test_data.province,'region_1':test_data.region_1,'region_2':test_data.region_2,'winery':test_data.winery,'variety':predicted_value})
output.to_csv('sample_submission1.csv',index=False)

#===============================================================>
#visualise the feature importance graph and confusion matrix 
#===============================================================>

import matplotlib.pyplot as plt
from xgboost import plot_importance
plot_importance(classifier)
plt.show()

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test1,y_test1_predict)
import seaborn as sns
plt.figure(figsize=(8,8))
sns.set(font_scale=1.4)
sns.heatmap(cm,square=True)
plt.xlabel('true value')
plt.ylabel('predicted value')
plt.show()

