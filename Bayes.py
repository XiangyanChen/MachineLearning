# coding:utf-8
import numpy as np
class Bayes:
    def train(self, dataSet, labels):   #训练样本
        self.vocabList = self.createVocabList(dataSet)  #创建特征词汇表
        trainMatrix = []    #多条词条向量的矩阵（一个词条向量代表着一个样本在词条中出现的次数）
        for line in dataSet:    #将每个训练样本转换为词条向量
            trainMatrix.append(self.setOfWord2Vec(self.vocabList, line))
        n = len(self.vocabList) #词条的特征数
        pN1Num = np.zeros(n)    #在类别为-1时，出现特征的次数向量（N1 = negative 1）
        p1Num = np.zeros(n)
        numOfPN1 = 0    #标签中出现-1的次数
        numOfP1 = 0
        for i in range(len(trainMatrix)):
            if labels[i] == 1:
                p1Num += trainMatrix[i] #与词条向量相加
                print(trainMatrix[i])
                numOfP1 += 1
            else:
                pN1Num += trainMatrix[i]
                numOfPN1 += 1
            # print trainMatrix[i]
        self.p1Vect =  p1Num / numOfP1   #p1的各个随机向量（特征）的概率分布
        self.pN1Vect = pN1Num / numOfPN1
        self.pClass1 = numOfP1 / float(len(labels)) #p(y=1)的概率
        return self.p1Vect, self.pN1Vect, self.pClass1

    def predict(self, inputData):   #预测函数
        inputVec = self.setOfWord2Vec(self.vocabList, inputData)#测试样本的词条向量
        # np.multiply(self.p1Vect ,inputVec)
        p1 = self.pClass1   #按照公式需要乘以p(y=1)的值，我们就以此为初始值
        pN1 = (1 - self.pClass1)
        for num in np.multiply(self.p1Vect ,inputVec):  #概率分布和词条向量进行相乘，得出p(x=xi|y=1)的概率，然后相乘
            if (num > 0):
                p1 *= num
        for num in np.multiply(self.pN1Vect ,inputVec):
            if (num > 0):
                pN1 *= num
        print(p1, pN1)
        if (p1 > pN1):  #相比，谁大就倾向谁
            return 1
        else:
            return -1
        # pOfLabels = {}
        # numOfLabels = len(labels)
        # labelSet = list(set(self.labels))
        # for i in range(len(labelSet)):  #遍历算出标签的概率
        #     count = 0
        #     for j in range(numOfLabels):
        #         if (labelSet[i] == labels[j]):
        #             count += 1
        #         pOfLabels[labelSet[i]] = count / float(numOfLabels)
        # print pOfLabels[-1]

    def createVocabList(self, dataSet): #创建词汇表
        vocabSet = set([])
        print(dataSet)
        for document in dataSet:
            print(document)
            vocabSet = vocabSet | set(document)
        return list(vocabSet)

    def setOfWord2Vec(self, vocabList, inputSet):   #词汇表向量
        returnVec = [0] * len(vocabList)    #vocablist大小的零向量
        for word in inputSet:   #遍历输入样本的每个特征
            if word in vocabList:
                returnVec[vocabList.index(word)] = 1    #如果发现有匹配的值就设置为1
        return returnVec

if __name__ == "__main__":
    dataSet = [[1, "S"], [1, "M"], [1, "M"], [1, "S"], [1, "S"],
               [2, "S"], [2, "M"], [2, "M"], [2, "L"], [2, "L"],
               [3, "L"], [3, "M"], [3, "M"], [3, "L"], [3, "L"]]
    labels = [-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1]
    bayes = Bayes()
    bayes.train(dataSet, labels)
    print(bayes.predict([2, "S"]))
    # print bayes.createVocabList(dataSet)