import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.ensemble import RandomForestRegressor


weight = 0.4
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
test_x = data_test.drop(['charges'], axis=1)

x1 = x.drop(['sex'], axis=1)
test_x1 = test_x.drop(['sex'], axis=1)
x1 = x1.drop(['region'], axis=1)
test_x1 = test_x1.drop(['region'], axis=1)
print('PolynomialFeatures-LinearRegression')
quad = PolynomialFeatures (degree=2)
x_quad = quad.fit_transform(x1)
plr = LinearRegression().fit(x_quad, y)
y_pred = plr.predict(x_quad)
print('R-square: %.4f' %r2_score(y, y_pred))
test_quad = quad.fit_transform(test_x1)
test_pred1 = plr.predict(test_quad)

print('------------------------------------------------------------')
print('RandomForestRegression')
# x = data.drop(['sex'], axis=1)
# test_x = data_test.drop(['sex'], axis=1)
# x = data.drop(['region'], axis=1)
# test_x = data_test.drop(['region'], axis=1)
forest = RandomForestRegressor(n_estimators=500, criterion='mse', n_jobs=-1)
forest.fit(x, y)
forest_train_pred = forest.predict(x)
print('MSE train data: %.4f' %mean_squared_error(y, forest_train_pred))
print('R-square: %.4f' %r2_score(y, forest_train_pred))
# test_pred2 = forest.predict(test_x)
test_pred2 = forest.predict(test_x) - bias
#
test_pred2 = weight * test_pred1 + (1 - weight) * test_pred2
data_test.charges = test_pred2
data_test.to_csv(test_root, index=False)