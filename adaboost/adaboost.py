import numpy as np

class SingleDecisionTree:   #决策树桩
    def __init__(self, axis=0, threshold = 0, flag = True):
        self.axis = axis
        self.threshold = threshold
        self.flag = flag #flag=True, x>=threshold=1, 否则为-1

    def preditct(self, x):
        if (self.flag == True):
            return -1 if x[self.axis] >= self.threshold else 1
        else:
            return 1 if x[self.axis] >= self.threshold else -1

    def preditctArr(self, dataSet):
        result = list()
        for x in dataSet:
            if (self.flag == True):
                result.append(-1 if x[self.axis] >= self.threshold else 1)
            else:
                result.append(1 if x[self.axis] >= self.threshold else -1)
        return result

class Adaboost:
    def predict(self, x):   #预测方法
        sum = 0
        for fun in self.funList:    #书上最终分类器的代码
            alpha = fun[0]
            tree = fun[1]
            sum += alpha * tree.preditct(x)
        return 1 if sum > 0 else -1

    def calcEm(self, D, Gm, dataSet, labels):    #计算误差
        # value = list()
        value = [0 if Gm.preditct(row) == labels[i] else 1 for (i, row) in enumerate(dataSet)]
        return np.sum(np.multiply(D, np.array(value).reshape((-1, 1))))

if __name__ == '__main__':
    # dataSet = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]  #例8.1的数据集
    # labels = [1, 1, 1, -1, -1, -1, 1, 1, 1, -1]
    dataSet = [[0, 1, 3], [0, 3, 1], [1, 2, 2], [1, 1, 3], [1, 2, 3], [0, 1, 2], [1, 1, 2], [1, 1, 1], [1, 3, 1], [0, 2, 1]]    #p153的例子
    labels = [-1, -1, -1, -1, -1, -1, 1, 1, -1, -1]
    # dataSet = [[1, 2], [1, 3], [2, 4], [3, 3], [3, 4], [4, 0.5], [5, 1.5], [5, 3], [5, 4]]    #练习题第一题的例子
    # labels = [-1, -1, -1, 1, 1, -1, 1, 1, 1]
    adaboost = Adaboost()
    adaboost.train(dataSet, labels)
    for x in dataSet:
        print(adaboost.predict(x))
    # print(adaboost.predict([1, 3, 2]))
