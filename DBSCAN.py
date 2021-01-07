import numpy as np
import math
import matplotlib.pyplot as plt


class dbscan():
    def __init__(self, data):
        self.data = data
        self.sample_num = data.shape[0]
        self.feat_num = data.shape[1]

    # 计算a和b之间的欧氏距离
    def eucli_dist(self, a, b):
        c = np.array(a)
        d = np.array(b)
        dis = math.sqrt(np.power(a - b, 2).sum())
        return dis

    # 计算闵可夫斯基距离
    def mink_dist(self, a, b, r):
        c = np.array(a)
        d = np.array(b)
        dis = math.pow(np.power(np.abs(c - d), r).sum(), 1.0 / r)
        return dis

    # 计算a和b之间的余弦距离
    def cosin_dist(self, a, b):
        c = np.array(a)
        d = np.array(b)
        dis = (c * d).sum() / (math.sqrt(np.power(c, 2).sum() * np.power(d, 2).sum()))
        return dis

    def dist(self, a, b):
        return self.eucli_dist(a, b)
        # return self.mink_dist(a, b, 3)

    # 确定eps和minPts
    # 记录每个点到其他点之间的距离并排序
    def select_para(self, k=4):
        print("calculating distances...")
        data = self.data
        d = []
        ave_dis = []
        for i in range(self.sample_num):
            if i % 100 == 0:
                print("%d/%d" %(i + 1, self.sample_num))
            dis_i = []
            data_i = np.array(data.iloc[i])
            for j in range(self.sample_num):
                data_j = np.array(data.iloc[j])
                dis_i.append(self.dist(data_i, data_j))
            dis_i.sort()
            ave_dis.append(dis_i[k - 1])
        ave_dis.sort(reverse=True)
        '''
        dis_diff = [0] * len(ave_dis)
        for i in range(1, len(ave_dis)):
            dis_diff[i] = ave_dis[i] - ave_dis[i - 1]
        max_diff = 0
        j = 0
        flag = False
        for i in range(1, len(dis_diff)):
            if dis_diff[i] - dis_diff[i - 1] > max_diff:
                max_diff = dis_diff[i] - dis_diff[i - 1]
                j = i
                flag = True
            elif flag == True:
                break
        print(j)
        '''
        X = np.arange(self.sample_num)
        plt.plot(X, ave_dis, 'r--')
        plt.ylabel("4th Nearest Neighbor Diatance")
        plt.xlabel("Points Sorted According to the 4th Nearest Neighbor")
        plt.show()
        return ave_dis

    def is_neighbor(self, a, b, eps):
        return self.dist(a, b) < eps

    # DBSCAN算法
    # 节点的eps半径范围内至少存在minPts个点
    def cluster(self, eps, minPts):
        data = self.data
        # 样本数
        data_num = self.sample_num
        noise_num = data_num
        # 初始化未访问过的点集
        unvisited = set(i for i in range(data_num))
        # 将所有点都标记为noise point(-1)
        type = [-1 for _ in range(data_num)]
        # n为各聚类的标签，初始值为-1
        n = -1
        # core point集合
        cores = []
        # 聚类集合
        clusters = []

        while unvisited:
            i = unvisited.pop()
            # 找出点i的所有neighbor
            neighbors = []
            core = data.iloc[i]
            for j in range(data_num):
                a = data.iloc[j]
                if self.is_neighbor(core, a, eps):
                    neighbors.append(j)

            # 如果点i的eps半径内所含节点数量不少于minPts，则点i为core point
            if len(neighbors) >= minPts:
                n += 1
                type[i] = n
                cores.append(i)
                cluster = [i]
                noise_num -= 1
                # 将neighbor加入该聚类
                for neighbor in neighbors:
                    if neighbor in unvisited:
                        type[neighbor] = n
                        unvisited.remove(neighbor)
                        border = data.iloc[neighbor]
                        cluster.append(neighbor)
                        border_neighbors = []
                        for k in range(data_num):
                            b = data.iloc[k]
                            if self.is_neighbor(border, b, eps):
                                border_neighbors.append(k)
                        # 将border points加入聚类
                        if len(border_neighbors) >= minPts:
                            for k in border_neighbors:
                                if k in unvisited:
                                    type[k] = n
                                    cluster.append(k)
                                    unvisited.remove(k)
                clusters.append(cluster)
            # 否则点i为noise point
            else:
                type[i] = -1
        # 返回每个点的标签，core point集合以及聚类列表
        # cores和clusters中存储的是样本的下标
        return type, cores, clusters
