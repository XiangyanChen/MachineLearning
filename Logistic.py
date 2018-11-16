# --*-- coding:utf-8 --*--
import numpy as np

class Logistic:
    def loadDataSet(self, fileName = 'files/logistic/dataset.txt'):   #加载数据
        dataMat = []
        labelMat = []
        fr = open(fileName)
        for line in fr.readlines(): #遍历文件
            lineArr = line.strip().split()
            dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])]) #数据集
            labelMat.append(int(lineArr[-1]))   #类别标签
        return dataMat, labelMat

    def sigmoid(self, inX):
        return 1.0 / (1 + np.exp(-inX))

    def train(self, dataSet, labels):   #训练
        dataMat = np.mat(dataSet)   #将数据集转成矩阵的形式
        labelMat = np.mat(labels).transpose()#将类别集合转成矩阵的形式
        m, n = np.shape(dataSet)    #行列
        alpha = 0.01
        maxIter = 500
        weights = np.ones((n, 1))
        for i in range(maxIter):    #迭代
            h = self.sigmoid(dataMat * weights)
            error = h - labelMat    #预测值和标签值所形成的误差
            weights = weights -  alpha * dataMat.transpose() * error    #权重的更新
        return weights

if __name__ == '__main__':
    logistic = Logistic()
    dataSet, labels = logistic.loadDataSet()
    weights = logistic.train(dataSet, labels)
    print(weights)



