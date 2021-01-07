import pandas as pd
import numpy as np
from Data import *
from DBSCAN import *
from PCA import *
import math
import matplotlib.pyplot as plt


# 计算聚类结果的所有聚类关于各自core_point的charges的平均方差
def getCSE(y, cores, clusters):
    cse = 0.0
    for i in range(len(cores)):
        core_charge = y[cores[i]]
        for j in clusters[i]:
            neighbor_charge = y[j]
            cse += (core_charge - neighbor_charge) * (core_charge - neighbor_charge)
    return cse / len(y)

def eval(data, target):
    x = data.drop([target], axis=1)
    y = data[target]
    dim = x.columns.__len__()
    # p = pca(data, dim)
    # data1 = p.reduce_dim()
    data1 = x
    d = dbscan(data1)
    ave_dis = d.select_para()

    # 由图可确定minPts的合理取值范围，遍历选取表现最好的情形
    eps0 = 1.25
    minPts0 = 29
    label0, cores0, clusters0 = [], [], []
    cse0 = math.inf
    X, Y = [], []
    for minPts in range(dim * 2, dim * 6):
        label, cores, clusters = d.cluster(eps0, minPts)
        cse = getCSE(y, cores, clusters)
        X.append(minPts)
        Y.append(cse)
        print("-" * 100)
        print("eps: ", eps0, " minPts: ", minPts)
        print("the number of clusters: ", len(cores))
        print("the CSE of DBSCAN: ", cse)
        if cse < cse0:
            label0, cores0, clusters0 = label, cores, clusters
            minPts0 = minPts
            cse0 = cse

    plt.plot(X, Y, 'b-')
    plt.ylabel("Mean Cluster Squared Error")
    plt.xlabel("minPts(eps = 1.25)")
    plt.show()
    print()
    print("=" * 100)
    print("the best condition with 3 clusters")
    print("eps: ", eps0, " minPts: ", minPts0)
    print("the CSE of DBSCAN: ", cse0)
    return label0, cores0, clusters0


if __name__ == '__main__':
    train_root = './public_dataset/train.csv'
    train_data_loader = data_loader(train_root)
    mean, std = train_data_loader.forward()
    train_data = train_data_loader.data
    # target = "charges"
    target = train_data.columns[-1]
    x = train_data.drop([target], axis=1)
    y = train_data[target]

    d = dbscan(x)

    # 由图可确定minPts的合理取值范围，遍历选取表现最好的情形
    eps0 = 1.25
    minPts0 = 37
    label, cores, clusters = d.cluster(eps0, minPts0)
    print("------------------------训练集聚类已完成------------------------")
    cores_charges = []
    for i in cores:
        cores_charges.append(y[i])
    cores_charges = np.array(cores_charges)

    test_root = './public_dataset/submission.csv'
    test_data_loader = data_loader(test_root)
    test_data_loader.forward(False, mean, std)
    test_data = test_data_loader.data
    test_data0 = pd.read_csv(test_root)

    x1 = test_data.drop([target], axis=1)
    test_num = x1.shape[0]
    test_pred = []
    for i in range(test_num):
        distances = []
        for j in cores:
            test_feat = np.array(x1.iloc[i])
            core_feat = np.array(x.iloc[j])
            dis = math.sqrt(np.power(test_feat - core_feat, 2).sum())
            distances.append(math.exp(-dis))
        distances = np.array(distances)
        distances = distances / distances.sum()
        test_pred.append((distances * cores_charges).sum())
        if i % 5 == 0:
            print(i)
    test_pred = np.array(test_pred)
    test_data0.charges = test_pred
    test_data0.to_csv(test_root, index=False)