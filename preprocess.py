import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder
import seaborn as sb


# calculate the correlations between charges and the others
def correlation(corr):
    print("属性间的相关性")
    print(corr['charges'].sort_values())
    f, ax = plt.subplots(figsize=(8, 8))
    sb.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool),
                cmap=sb.diverging_palette(240,10,as_cmap=True),square=True, ax=ax)
    f.show()

def distribution(data, attr):
    f = plt.figure(figsize=(12, 5))

    ax = f.add_subplot(121)
    sb.distplot(data[(data[attr] == 1)]["charges"], color='c', ax=ax)
    ax.set_title('Distribution of charges for smokers')

    ax = f.add_subplot(122)
    sb.distplot(data[(data[attr] == 0)]['charges'], color='b', ax=ax)
    ax.set_title('Distribution of charges for non-smokers')
    f.show()


data = pd.read_csv('public_dataset/train.csv')
print(data.shape)
print(data.count())
print(data.head())
print(data.isnull().sum())

data_type = pd.DataFrame({'columns':data.columns.tolist(),
                          'type':data.dtypes.tolist(),
                          'example0':data.iloc[0].tolist()})
print(data_type)

des = data['charges'].describe()
print(des)

# sex -> 0/1
le = LabelEncoder()
le.fit(data.sex.drop_duplicates())
data.sex = le.transform(data.sex)
# smoker -> 0/1
le.fit(data.smoker.drop_duplicates())
data.smoker = le.transform(data.smoker)
# region -> 0/1/2/3
le.fit(data.region.drop_duplicates())
data.region = le.transform(data.region)
print(data.head())

correlation(data.corr())
distribution(data, "smoker")