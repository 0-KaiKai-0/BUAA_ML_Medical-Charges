import pandas as pd
import numpy as np
from Data import *
from DBSCAN import *
from PCA import *
import math
import matplotlib.pyplot as plt


# # 将所有noise_point分类为离他最近的core_point一类
# def merge_noise(d, label1, cores):
#     for i in range(len(label1)):
#         if label1[i] == -1:
#             l = label1[cores[0]]
#             dis = d.dist(i, 0)
#             j = 1
#             while j < len(cores):
#                 dis_ij = d.dist(i, j)
#                 if dis > dis_ij:
#                     dis = dis_ij
#                     l = label1[cores[j]]
#                 j += 1
#             label1[i] = l
#     return label1

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
    minPts0 = 0
    label0, cores0, clusters0 = [], [], []
    cse0 = math.inf
    X, Y = [], []
    for minPts in range(dim * 6, dim * 10):
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
    print("the best condition:")
    print("eps: ", eps0, " minPts: ", minPts0)
    print("the CSE of DBSCAN: ", cse0)
    return label0, cores0, clusters0


if __name__ == '__main__':
    train_root = './public_dataset/train.csv'
    train_data_loader = data_loader(train_root)
    train_data_loader.forward()
    train_data = train_data_loader.data
    # target = "charges"
    target = train_data.columns[-1]
    label0, cores0, clusters0 = eval(train_data, target)

    test_root = './public_dataset/submission.csv'
    test_data_loader = data_loader(test_root)
    test_data_loader.forward()
    test_data = test_data_loader.data