# --*-- coding:utf-8 --*--
import numpy as np

class Logistic:
    def load_dataSet(self, fileName = 'dataset.txt'):   #加载数据
        data_mat = []
        label_mat = []
        fr = open(fileName)
        for line in fr.readlines(): #遍历文件
            line_arr = line.strip().split()
            data_mat.append([1.0, float(line_arr[0]), float(line_arr[1])]) #数据集
            label_mat.append(int(line_arr[-1]))   #类别标签
        return data_mat, label_mat

    def sigmoid(self, inX):
        return 1.0 / (1 + np.exp(-inX))

    def train(self, dataSet, labels):   #训练
        data_mat = np.mat(dataSet)   #将数据集转成矩阵的形式
        label_mat = np.mat(labels).transpose()#将类别集合转成矩阵的形式
        m, n = np.shape(dataSet)    #行列
        alpha = 0.01
        max_iter = 500
        weights = np.ones((n, 1))
        for i in range(max_iter):    #迭代
            h = self.sigmoid(data_mat * weights)
            error = h - label_mat    #预测值和标签值所形成的误差
            weights = weights -  alpha * data_mat.transpose() * error    #权重的更新
        return weights

if __name__ == '__main__':
    logistic = Logistic()
    dataSet, labels = logistic.load_dataSet()
    weights = logistic.train(dataSet, labels)
    print(weights)



