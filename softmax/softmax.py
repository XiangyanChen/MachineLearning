import numpy as np
import os

class Softmax:
    def loadData(self, dir):    #给出文件目录，读取数据
        digits = list() #数据集（数据）
        labels = list() #标签
        if os.path.exists(dir): #判断目录是否存在
            files = os.listdir(dir) #获取目录下的所有文件名
            for file in files:  #遍历所有文件
                labels.append(file.split('_')[0])   #按照文件名规则，文件名第一位是标签
                with open(dir + '\\' + file) as f:  #通过“目录+文件名”，获取文件内容
                    digit = list()
                    for line in f:  #遍历文件每一行
                        digit.extend(map(int, list(line.replace('\n', ''))))    #遍历每行时，把数字通过extend的方法扩展
                    digits.append(digit)    #将数据扩展进去
        digits = np.array(digits)   #数据集
        labels = list(map(int, labels)) #标签
        labels = np.array(labels).reshape((-1, 1))  #将标签重构成(N, 1)的大小
        return digits, labels

    def softmax(self, X):   #softmax函数
        return np.exp(X) / np.sum(np.exp(X))

    def train(self, digits, labels, maxIter = 100, alpha = 0.1):
        self.weights = np.random.uniform(0, 1, (10, 1024))
        for iter in range(maxIter):
            for i in range(len(digits)):
                x = digits[i].reshape(-1, 1)
                y = np.zeros((10, 1))
                y[labels[i]] = 1
                y_ = self.softmax(np.dot(self.weights, x))
                self.weights -= alpha * (np.dot((y_ - y), x.T))
        return self.weights

    def predict(self, digit):   #预测函数
        return np.argmax(np.dot(self.weights, digit))   #返回softmax中概率最大的值

if __name__ == '__main__':
    softmax = Softmax()
    trainDigits, trainLabels = softmax.loadData('dataset/trainingDigits')
    testDigits, testLabels = softmax.loadData('dataset/testDigits')
    softmax.train(trainDigits, trainLabels, maxIter=100) #训练
    accuracy = 0
    N = len(testDigits) #总共多少测试样本
    for i in range(N):
        digit = testDigits[i]   #每个测试样本
        label = testLabels[i][0]    #每个测试标签
        predict = softmax.predict(digit)  #测试结果
        if (predict == label):
            accuracy += 1
        print("predict:%d, actual:%d"% (predict, label))
    print("accuracy:%.1f%%" %(accuracy / N * 100))