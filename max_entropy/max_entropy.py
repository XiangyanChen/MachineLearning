# --*-- coding:utf-8 --*--
import numpy as np

class MaxEntropy:
    def __init__(self):
        self.weights = None
        self.feature_dict = None
        pass

    def train(self, dataset, labels, max_iter = 1000):
        #将样本集([1, "M"], [1, "M"], [1, "M"])和标签集[-1, -1, 1] ->{[1, "M"]:{-1:2, 1:1}}
        feature_dict = dict()   #特征函数->事实的集合，比如有数据，f = (x, y) = ([1, 2],1) ->是1
        n = len(dataset)    #样本数目
        alpha = 0.01    #学习率
        for data, label in zip(dataset, labels):    #同时遍历两个列表
            data = str(data)
            if (feature_dict.get(data) == None): #{-1:2, 1:1}
                feature_dict[data] = dict()
            label_dict = feature_dict.get(data)
            if (label_dict.get(label) == None):
                label_dict[label] = (1, 0)  #(count, weight)
            else:
                count = label_dict[label][0]
                weight = label_dict[label][1]
                label_dict[label] = (count+1, weight)
            # label_dict[label] = label_dict.get(label, 0) + 1    #自增
        self.feature_dict = feature_dict
        for i in range(max_iter):
            for data, label_dict in self.feature_dict.items():
                P_marginal_empirical_x = 0
                for label, count_weight_tuple in label_dict.items():
                    count = count_weight_tuple[0]
                    P_marginal_empirical_x += count / n
                for label, count_weight_tuple in label_dict.items():
                    count = count_weight_tuple[0]
                    weight = count_weight_tuple[1]
                    new_weight = weight - alpha * (P_marginal_empirical_x * count * self.predict(data, label) - count / n * count)
                    print(new_weight)
                    self.feature_dict[data][label] = (count, new_weight)
        print(feature_dict)


    def predict(self, data, label):#预测label=1的概率
        data = str(data)
        numerator = 0   #分子
        denominator = 0 #分母
        for key, count_weight_tuple in self.feature_dict[data].items():   #遍历{-1: (2, 0), 1: (1, 0)}
            count = count_weight_tuple[0]   # 2
            weight = count_weight_tuple[1]   # 0
            if (key == label):
                numerator = np.exp(weight * count)
            denominator += np.exp(weight * count)
        return numerator / denominator



    # def __init__(self):
    #     self.w = np.zeros(10)
    #     self.n = 0
    #     self.M = 1000
    #     return
    # def train(self, dataSet, labels, maxIter = 1000):    #训练
    #     self.dataSet = dataSet
    #     featureList = set()
    #     featureCount = {}
    #     self.y = set()
    #     for i, row in enumerate(dataSet):
    #         for axis, x in enumerate(row):
    #             featureList.add((axis, x, labels[i]))
    #             featureCount[(axis, x, labels[i])] = featureCount.get((axis, x, labels[i]), 0) + 1
    #             featureCount[(axis, x)] = featureCount.get((axis, x), 0) + 1
    #             self.y.add(labels[i])
    #     self.y = list(self.y)
    #     self.featureList = list(featureList)
    #     self.featureCount = featureCount
    #     self.n = len(featureList)
    #     self.w = np.zeros(self.n)
    #     delta = np.zeros(self.n)
    #     for k in range(maxIter):
    #         for i in range(self.n):
    #             delta[i] = 1 / float(self.M) * (np.log(self.clacETildeP(self.featureList[i]) / float(self.calcEP(self.featureList[i]))))
    #             self.w += delta[i]
    #     return self.w
    #
    # def calcEP(self, feature):
    #     return float(self.featureCount[(feature[0], feature[1])]) * float(self.featureCount[(feature[0], feature[1], feature[2])])
    # def clacETildeP(self, feature):
    #     return self.featureCount[(feature[0], feature[1], feature[2])]
    #
    # def predict(self, data):  #预测
    #     numerator = 0
    #     denominator = 0
    #     exp = np.zeros(len(self.y))   #P88的每个exp的含义
    #     for index, y in enumerate(self.y):
    #         f = np.zeros(self.n)
    #         for i in range(self.n):
    #             for index, x in enumerate(data):
    #                 if self.featureCount[(index, x, y)] > 0:
    #                     f[i] = 1
    #         exp[index] = sum(self.w.transpose() * f)
    #     max = 0
    #     for x in exp:
    #         print(x)
    #         if (max < x):
    #             max = x
    #     return max
if __name__ == '__main__':
    dataSet = [[1, "S"], [1, "M"], [1, "M"], [1, "S"], [1, "S"],
               [2, "S"], [2, "M"], [2, "M"], [2, "L"], [2, "L"],
               [3, "L"], [3, "M"], [3, "M"], [3, "L"], [3, "L"]]
    labels = [-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1]
    maxEntropy = MaxEntropy()
    maxEntropy.train(dataSet, labels)
    print(maxEntropy.predict([1, "M"], 1))
