# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 14:48:14 2019

@author: dell
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 16:10:23 2019

@author: dell
"""

import pandas as pd, numpy as np, re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.svm import LinearSVR
import statsmodels.api as sm
from keras.models import Sequential
from keras.layers import Dense
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor


data = pd.read_excel('Data_Train.xlsx')

def is_null(data):
    print(data.isnull().any(axis = 0))
    
is_null(data)

data = data.dropna()

is_null(data)

def info(data):
    print(data.info())

info(data)

def unique(data, column):
    print(data[column].unique())

unique(data, 'RATING')
unique(data, 'VOTES')

data['VOTES'] = data['VOTES'].apply(lambda v : v.split()[0]).astype(np.float64)

data['RATING'] = data['RATING'].astype(np.float64)

info(data)

print(data.head())

#def remove_outlier(data):
#    boxplot = data.boxplot(return_type='dict')
#    fliers = [flier.get_ydata() for flier in boxplot['fliers']]
#    data = data[data['VOTES'].apply(lambda v: True if v not in fliers[2] else False)]
#    data = data[data['COST'].apply(lambda v: True if v not in fliers[3] else False)]
#    return data
#    
#data = remove_outlier(data)    

print(data.columns)
print(data.head())

def features_and_labels(data):
    features = data.drop(['RESTAURANT_ID', 'TIME', 'COST'], axis = 1)
    labels = data['COST']
    return features, labels

features, labels = features_and_labels(data)

def feature_engg(features):
    title = features['TITLE'].apply(lambda t: ' '.join(t.split(','))).values
    cuisines = features['CUISINES'].apply(lambda c: ' '.join(c.split(', '))).values
    
    features = features.drop(['TITLE', 'CUISINES'], axis = 1).values
    
    return title, cuisines, features

title, cuisines, features = feature_engg(features)

cv1 = CountVectorizer(max_features = 2000)
title = cv1.fit_transform(title).toarray()

cv2 = CountVectorizer(max_features = 2000)
cuisines = cv2.fit_transform(cuisines).toarray()

cv3 = CountVectorizer(max_features = 2000)
city = cv3.fit_transform(features[:, 0]).toarray()

cv4 = CountVectorizer(max_features = 2000)
locality = cv4.fit_transform(features[:, 1]).toarray()
#
#le1 = LabelEncoder()
#le2 = LabelEncoder()
#
#features[:, 0] = le1.fit_transform(features[:, 0])
#features[:, 1] = le2.fit_transform(features[:, 1])
#    
#ohe = OneHotEncoder(categorical_features = [0, 1])
#features = ohe.fit_transform(features).toarray()    





features = np.concatenate((title, cuisines, city, locality, features[:, [2, 3]]), axis = 1)

# spliting training and testing data
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.01, random_state = 0)

# model 1 :-
lr = LinearRegression()
lr.fit(features_train, labels_train)
print(lr.score(features_train, labels_train))
print(lr.score(features_test, labels_test))

labels_pred = lr.predict(features_test)
print(pd.DataFrame(zip(labels_test.values, labels_pred)))

# model 2 :-
lasso = Lasso()
lasso.fit(features_train, labels_train)
print(lasso.score(features_train, labels_train))
print(lasso.score(features_test, labels_test))

# model 3 :-
ridge = Ridge()
ridge.fit(features_train, labels_train)
print(ridge.score(features_train, labels_train))
print(ridge.score(features_test, labels_test))

# model 4 :-
en = ElasticNet()
en.fit(features_train, labels_train)
print(en.score(features_train, labels_train))
print(en.score(features_test, labels_test))

# model 5 :-
dtr = DecisionTreeRegressor()
dtr.fit(features_train, labels_train)
print(dtr.score(features_train, labels_train))
print(dtr.score(features_test, labels_test))

# model 6 :-
rfr = RandomForestRegressor(n_estimators = 50, random_state = 0)
rfr.fit(features_train, labels_train)
print(rfr.score(features_train, labels_train))
print(rfr.score(features_test, labels_test))

# model 7 :-
svr = SVR()
svr.fit(features_train, labels_train)
print(svr.score(features_train, labels_train))
print(svr.score(features_test, labels_test))

# model 8 :-
lsvr = LinearSVR()
lsvr.fit(features_train, labels_train)
print(lsvr.score(features_train, labels_train))
print(lsvr.score(features_test, labels_test))

# model 9 :-
knn = KNeighborsRegressor(n_neighbors = 5)
knn.fit(features_train, labels_train)
print(knn.score(features_train, labels_train))
print(knn.score(features_test, labels_test))

# deep learning model :-
model = Sequential()
model.add(Dense(units = 100, kernel_initializer = 'uniform', activation = 'relu', input_dim = 1797))
model.add(Dense(units = 100, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 1, kernel_initializer = 'uniform'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.fit(features_train, labels_train, batch_size = 32, epochs = 10)








# dealing with test data
test_data = pd.read_excel('Data_Test.xlsx')
info(test_data)

is_null(test_data)

unique(test_data, 'CITY')
unique(test_data, 'LOCALITY')
unique(test_data, 'RATING')
unique(test_data, 'VOTES')

test_data['RATING'] = test_data['RATING'].replace(['-', 'NEW'], '0').astype(np.float64)
test_data['RATING'] = test_data['RATING'].fillna(0)

test_data['VOTES'] = test_data['VOTES'].fillna('0 votes')

test_data['VOTES'] = test_data['VOTES'].apply(lambda v : v.split()[0]).astype(np.float64)

test_data['CITY'] = test_data['CITY'].fillna(test_data['CITY'].mode()[0])

test_data['LOCALITY'] = test_data['LOCALITY'].fillna(test_data['LOCALITY'].mode()[0])

test_df = test_data.drop(['RESTAURANT_ID', 'TIME'], axis = 1)

title_test, cuisines_test, test_values =  feature_engg(test_df)

title_test = cv1.transform(title_test).toarray()
cuisines_test = cv2.transform(cuisines_test).toarray()
city_test = cv3.transform(test_values[:, 0]).toarray()
locality_test = cv4.transform(test_values[:, 1]).toarray()

test_features = np.concatenate((title_test, cuisines_test, city_test, locality_test, test_values[:, [2, 3]]), axis = 1)

result = rfr.predict(test_features)

result = pd.DataFrame(result, columns = ['COST'])

result.to_excel('Cost_Prediction.xlsx', index = False)