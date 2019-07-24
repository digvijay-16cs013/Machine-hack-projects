# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 13:57:31 2019

@author: dell
"""

import pandas as pd, numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor

data = pd.read_excel("Data_Train.xlsx")

# to check null values
def check_nan(df):
    print(df.isnull().any(axis = 0))

# to count null values in a column
def count_nan(col):
    print(len(col[col.isnull()]))

# to get the information of columns
def col_info(df):
    print(df.info())
    
# to get the unique values of column
def unique_values(col):
    print(col.unique())
    
# since "New_Price" column is having plethora of null values there dropping this column
data = data.drop(["Name", "New_Price"], axis = 1)

# checking columns with null values
check_nan(data)

# counting number of null values in different columns
count_nan(data['Mileage'])
count_nan(data['Engine'])
count_nan(data['Power'])
count_nan(data['Seats'])

# checking unique_values
unique_values(data['Mileage'])
unique_values(data['Engine'])
unique_values(data['Power'])
unique_values(data['Seats'])


col_info(data)

print(data['Mileage'].head())
print(data['Engine'].head())
print(data['Power'].head())

# filling nan values with mode(most appeared) value
data = data.fillna(data.mode().iloc[0, :])

# now checking for null values
check_nan(data)

data['Mileage'] = data['Mileage'].apply(lambda m: m.split()[0]).astype(np.float64)
data['Engine'] = data['Engine'].apply(lambda e: e.split()[0]).astype(np.float64)

# here we encountered an error and, after observing we got that there is a value "null bhp" in "Power" Column
# Run this statement to check for error
# data['Power'] = data['Power'].apply(lambda p: p.split()[0]).astype(np.float64)

# To fix the "null bhp" (Replacing it with the most common value i.e. mode value)
data['Power'] = data['Power'].replace('null bhp', data['Power'].mode()[0])

# now running the statement again
data['Power'] = data['Power'].apply(lambda p: p.split()[0]).astype(np.float64)

col_info(data)

data.boxplot()

boxplot = data.boxplot(return_type='dict')

# To remove outliers
#outliers = [f.get_ydata() for f in boxplot['fliers']]
#
#cols = ['Year', 'Kilometers_Driven', 'Mileage', 'Engine', 'Power', 'Seats', 'Price']
#
#for col, flier in zip(cols, outliers):
#    data = data[data[col].apply(lambda x: True if x not in flier else False)]

features = data.drop(['Price'], axis = 1).values
labels = data['Price'].values

le1 = LabelEncoder()
le2 = LabelEncoder()
le3 = LabelEncoder()
le4 = LabelEncoder()

features[:, 0] = le1.fit_transform(features[:, 0])
features[:, 3] = le2.fit_transform(features[:, 3])
features[:, 4] = le3.fit_transform(features[:, 4])
features[:, 5] = le4.fit_transform(features[:, 5])

ohe = OneHotEncoder(categorical_features=[0])
features = ohe.fit_transform(features).toarray()

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.01, random_state = 0)

# model 1:-

lr = LinearRegression()
lr.fit(features_train, labels_train)
print(lr.score(features_train, labels_train))
print(lr.score(features_test, labels_test))


# model 2:-

rfr = RandomForestRegressor(n_estimators = 23, random_state = 0)
rfr.fit(features_train, labels_train)
print(rfr.score(features_train, labels_train))
print(rfr.score(features_test, labels_test))


# model 3:-

knn = KNeighborsRegressor()
knn.fit(features_train, labels_train)
print(knn.score(features_train, labels_train))
print(knn.score(features_test, labels_test))


# model 4:-

svr = SVR()
svr.fit(features_train, labels_train)
print(svr.score(features_train, labels_train))
print(svr.score(features_test, labels_test))


# model 5:-

lsvr = LinearSVR()
lsvr.fit(features_train, labels_train)
print(lsvr.score(features_train, labels_train))
print(lsvr.score(features_test, labels_test))


# model 6:-

dtr = DecisionTreeRegressor()
dtr.fit(features_train, labels_train)
print(dtr.score(features_train, labels_train))
print(dtr.score(features_test, labels_test))

test_data = pd.read_excel('Data_Test.xlsx')

test_data = test_data.drop(['New_Price', 'Name'], axis = 1)
test_data = test_data.fillna(test_data.mode().iloc[0, :])

# checking for nan values in test data
check_nan(test_data)

test_data['Mileage'] = test_data['Mileage'].apply(lambda m: m.split()[0]).astype(np.float64)
test_data['Engine'] = test_data['Engine'].apply(lambda e: e.split()[0]).astype(np.float64)

# here we encountered an error and, after observing we got that there is a value "null bhp" in "Power" Column
# Run this statement to check for error
test_data['Power'] = test_data['Power'].apply(lambda p: p.split()[0]).astype(np.float64)

# To fix the "null bhp" (Replacing it with the most common value i.e. mode value)
test_data['Power'] = test_data['Power'].replace('null bhp', test_data['Power'].mode()[0])

# now running the statement again
test_data['Power'] = test_data['Power'].apply(lambda p: p.split()[0]).astype(np.float64)

col_info(test_data)

test_data['Location'] = le1.transform(test_data['Location'])
test_data['Fuel_Type'] = le2.transform(test_data['Fuel_Type'])
test_data['Transmission'] = le3.transform(test_data['Transmission'])
test_data['Owner_Type'] = le4.transform(test_data['Owner_Type'])

test_data = ohe.transform(test_data).toarray()

test_predict = pd.DataFrame(rfr.predict(test_data))

test_predict = test_predict[0].apply(lambda v: round(v, 2))
test_df = pd.DataFrame(data = test_predict)
test_df.columns = ['Price']
test_df.to_excel('Test_Results3.xlsx', index = False)

