import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.ensemble import RandomForestRegressor


random_seed = 5
weight = 0.6
bias = 300

train_root = './public_dataset/train.csv'
test_root = './public_dataset/submission.csv'
data = pd.read_csv(train_root)
data_test = pd.read_csv(test_root)

# sex -> 0/1
le = LabelEncoder()
le.fit(data.sex.drop_duplicates())
data.sex = le.transform(data.sex)
le.fit(data_test.sex.drop_duplicates())
data_test.sex = le.transform(data_test.sex)
# smoker -> 0/1
le.fit(data.smoker.drop_duplicates())
data.smoker = le.transform(data.smoker)
le.fit(data_test.smoker.drop_duplicates())
data_test.smoker = le.transform(data_test.smoker)
# region -> 0/1/2/3
le.fit(data.region.drop_duplicates())
data.region = le.transform(data.region)
le.fit(data_test.region.drop_duplicates())
data_test.region = le.transform(data_test.region)

x = data.drop(['charges'], axis=1)
y = data.charges

print('LinearRegression')
x_train1, x_test1, y_train1, y_test1 = train_test_split(x, y, random_state=random_seed)
lr = LinearRegression().fit(x_train1, y_train1)
y_test_pred1 = lr.predict(x_test1)
# print('score: %.4f' %lr.score(x_test1, y_test1))
print('R-square: %.4f' %r2_score(y_test1, y_test_pred1))

print('------------------------------------------------------------')
print('PolynomialFeatures-LinearRegression')
quad = PolynomialFeatures (degree=2)
x_quad = quad.fit_transform(x)
x_train2, x_test2, y_train2, y_test2 = train_test_split(x_quad, y, random_state=random_seed)
plr = LinearRegression().fit(x_train2, y_train2)
y_test_pred2 = plr.predict(x_test2)
# print('score: %.4f' %plr.score(x_test2, y_test2))
print('R-square: %.4f' %r2_score(y_test2, y_test_pred2))

print('------------------------------------------------------------')
print('RandomForestRegression')
x_train1 = x_train1.drop(['sex'], axis=1)
x_test1 = x_test1.drop(['sex'], axis=1)
x_train1 = x_train1.drop(['region'], axis=1)
x_test1 = x_test1.drop(['region'], axis=1)
forest1 = RandomForestRegressor(n_estimators=500, criterion='mse', n_jobs=-1)
forest1.fit(x_train1, y_train1)
forest_train_pred = forest1.predict(x_train1)
forest_test_pred = forest1.predict(x_test1)
print('MSE train data: %.3f, MSE test data: %.3f' % (
mean_squared_error(y_train1, forest_train_pred),
mean_squared_error(y_test1, forest_test_pred)))
print('R2 train data: %.3f, R2 test data: %.3f' %
      (r2_score(y_train1, forest_train_pred), r2_score(y_test1, forest_test_pred)))
b = forest_test_pred - bias
c = b * weight + y_test_pred2 * (1 - weight)
a = c > y_test1
print(a[a==True].size)
print(r2_score(y_test2, b))
