# --*-- coding:utf:8 --*--
import numpy as np

class PerceptronDual:  # 感知机
    def __init__(self, dataSet, labels):  # 初始化数据集和标签
        self._dataSet = np.array(dataSet)
        self._labels = np.array(labels).transpose()

    def train(self):
        m, n = np.shape(self.dataSet)  # m是行和n是列
        weights = np.zeros(n)
        bias = 0
        flag = False
        Gram = np.zeros((m, m))
        for i in range(m):  # 计算Gram矩阵 gram matrix
            for j in range(m):
                Gram[i][j] = dataSet[i] * np.mat(dataSet[j]).transpose()
        print(Gram)
        a = np.zeros(m)
        while flag != True:
            flag = True
            for i in range(m):  # 遍历样本
                sum = 0
                for j in range(m):  # 求误分条件
                    sum += a[j] * self.labels[j] * Gram[j][i]
                sum += bias
                if (sum * self.labels[i] <= 0):
                    a[i] += 1
                    bias += self.labels[i]
                    flag = False
        for i in range(m):
            weights += a[i] * self.dataSet[i] * self.labels[i]
        return weights, bias

    def sign(self, y):  # 符号函数
        if (y > 0):
            return 1
        else:
            return -1

    @property
    def dataSet(self):
        return self._dataSet

    @property
    def labels(self):
        return self._labels

if __name__ == '__main__':
    dataSet = [[3, 3],
               [4, 3],
               [1, 1]]
    labels = [1, 1, -1]
    perceptron = PerceptronDual(dataSet, labels)  # 创建一个感知机对象
    weights, bias = perceptron.train()  # 训练
    print("result is :%s, %s" % (weights, bias))