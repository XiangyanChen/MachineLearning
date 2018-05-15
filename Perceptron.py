# --*-- coding:utf:8 --*--
import numpy as np


class Perceptron:  # 感知机
    def __init__(self, dataSet, labels):  # 初始化数据集和标签
        self.dataSet = np.array(dataSet)
        self.labels = np.array(labels).transpose()

    def train(self):
        m, n = np.shape(self.dataSet)  # m是行和n是列
        weights = np.zeros(n)
        bias = 0
        flag = False
        while flag != True:
            flag = True
            for i in range(m):  # 遍历样本
                y = weights * np.mat(dataSet[i]).T + bias  # 以向量的形式计算
                # y = sum(weights * dataSet[i]) + bias
                if (self.sign(y) * self.labels[i] < 0):  # 说明是误分类了
                    weights += self.labels[i] * self.dataSet[i]  # 更新权重
                    bias += self.labels[i]  # 更新偏置
                    print("weights %s" % weights)
                    print("bias %s" % bias)
                    flag = False
        return weights, bias

    def sign(self, y):  # 符号函数
        if (y > 0):
            return 1
        else:
            return -1


if __name__ == "__main__":
    dataSet = [[3, 3],
               [4, 3],
               [1, 1]]
    labels = [1, 1, -1]
    perceptron = Perceptron(dataSet, labels)  # 创建一个感知机对象
    weights, bias = perceptron.train()  # 训练
    print("结果是:%s, %s" % (weights, bias))  