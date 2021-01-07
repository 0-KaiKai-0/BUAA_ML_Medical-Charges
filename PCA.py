import numpy as np

class pca():
    def __init__(self, data, dim):
        self.data = data
        self.dim = dim

        if dim >= data.shape[1]:
            print("target dimension must be smaller than initial dimension")

    def reduce_dim(self):
        sample_num = self.data.shape[0]
        feat_num = self.data.shape[1]
        mean = np.mean(self.data, 0)
        means = np.tile(mean, (sample_num, 1))
        # 将数据零均值化
        data = np.array(self.data - means).astype(float)
        # 求协方差矩阵
        cov = np.cov(data.T)
        # 求协方差矩阵的特征值与特征向量
        val, vec = np.linalg.eig(cov)
        # 对特征值进行排序，记录下标
        idx = np.argsort(-val)
        print("We select attributes", idx[:self.dim])
        #选择最大的dim个特征值所对应的特征向量，并转置
        vec1 = np.matrix(vec.T[idx[:self.dim]])
        return data * vec1.T