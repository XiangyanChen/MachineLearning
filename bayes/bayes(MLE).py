# coding:utf-8
import numpy as np

class Bayes:
    def train(self, dataSet, labels):   #训练样本 train
        self._vocab_list = self.create_vocab_list(dataSet)  #创建特征词汇表 create vocab list
        train_matrix = []    #多条词条向量的矩阵（一个词条向量代表着一个样本在词条中出现的次数）
        for line in dataSet:    #将每个训练样本转换为词条向量
            train_matrix.append(self.set_of_word2vec(self._vocab_list, line))
        n = len(self._vocab_list) #词条的特征数
        pn1_num = np.zeros(n)    #在类别为-1时，出现特征的次数向量（n1 means negative 1）
        p1_num = np.zeros(n)    #
        num_of_pn1 = 0    #标签中出现-1的次数
        num_of_p1 = 0
        for i in range(len(train_matrix)):
            if labels[i] == 1:
                p1_num += train_matrix[i] #与词条向量相加
                print(train_matrix[i])
                num_of_p1 += 1
            else:
                pn1_num += train_matrix[i]
                num_of_pn1 += 1
            # print trainMatrix[i]
        self._p1_vec =  p1_num / num_of_p1   #p1的各个随机向量（特征）的概率分布
        self._pn1_vec = pn1_num / num_of_pn1
        self._p_class_1 = num_of_p1 / float(len(labels)) #p(y=1)的概率
        return self._p1_vec, self._pn1_vec, self._p_class_1

    def predict(self, inputData):   #预测函数
        inputVec = self.set_of_word2vec(self._vocab_list, inputData)#测试样本的词条向量
        # np.multiply(self.p1Vect ,inputVec)
        p1 = self._p_class_1   #按照公式需要乘以p(y=1)的值，我们就以此为初始值
        p_n1 = (1 - self._p_class_1)
        for num in np.multiply(self._p1_vec ,inputVec):  #概率分布和词条向量进行相乘，得出p(x=xi|y=1)的概率，然后相乘
            if (num > 0):
                p1 *= num
        for num in np.multiply(self._pn1_vec ,inputVec):
            if (num > 0):
                p_n1 *= num
        print(p1, p_n1)
        if (p1 > p_n1):  #相比，谁大就倾向谁
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

    def create_vocab_list(self, dataSet): #创建词汇表 create vocab list, like[1, 2, 3, 'S', 'M', 'L']
        vocab_set = set()
        for document in dataSet:
            vocab_set = vocab_set | set(document)
        return list(vocab_set)

    def set_of_word2vec(self, vocab_list, input_set):   #词汇表向量
        return_vec = [0] * len(vocab_list)    #vocablist大小的零向量
        for word in input_set:   #遍历输入样本的每个特征
            if word in vocab_list:
                return_vec[vocab_list.index(word)] = 1    #如果发现有匹配的值就设置为1
        return return_vec

if __name__ == "__main__":
    dataSet = [[1, "S"], [1, "M"], [1, "M"], [1, "S"], [1, "S"],
               [2, "S"], [2, "M"], [2, "M"], [2, "L"], [2, "L"],
               [3, "L"], [3, "M"], [3, "M"], [3, "L"], [3, "L"]]
    labels = [-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1]
    bayes = Bayes()
    bayes.train(dataSet, labels)
    print(bayes.predict([2, "S"]))
    # print bayes.createVocabList(dataSet)