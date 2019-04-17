# coding:utf-8
import numpy as np

class Bayes:
    def create_vocab_list(self, dataSet): #创建词汇表 create vocab list, like[1, 2, 3, 'S', 'M', 'L']
        vocab_set = set()
        for document in dataSet:
            vocab_set = vocab_set | set(document)
        return list(vocab_set)

    def set_of_word2vec(self, vocab_list, input_set):   #词条向量 return feature vector, like a feature vector is [1, 'S']，if vocab_list is [1, 2, 3, 'S', 'M', 'L']，return vocab vector [1, 0, 0, 1, 0, 0]
        vocab_vec = [0] * len(vocab_list)    #vocablist大小的零向量 zero vector
        for word in input_set:   #遍历输入样本的每个特征   iterating every feature
            if word in vocab_list:
                vocab_vec[vocab_list.index(word)] = 1    #如果发现有匹配的值就设置为1
        return vocab_vec

    def train(self, dataSet, labels):   #训练样本 train
        self._vocab_list = self.create_vocab_list(dataSet)  #创建特征词汇表 create vocab list
        train_matrix = []    #多条词条向量的矩阵（一个词条向量代表着一个样本在词条中出现的次数） matrix consists of vocab vector
        for line in dataSet:    #将每个训练样本转换为词条向量 feature vector to vocab vector
            train_matrix.append(self.set_of_word2vec(self.vocab_list, line))
        n = len(self.vocab_list) #词条的特征数   feature num
        negative_feature_num = np.zeros(n)    #在类别为-1时，出现特征的次数向量（n1 means negative 1）,the vector of counting num of every feature when label equal -1
        positve_feature_num = np.zeros(n)    #在类别为1时，出现特征的次数向量（）
        negative_num = 0    #标签中出现-1的次数 counting the number of negative label
        positive_num = 0
        for i in range(len(train_matrix)):
            if labels[i] == 1:
                positive_num += 1
                positve_feature_num += train_matrix[i]
            else:
                negative_feature_num += train_matrix[i] #与词条向量相加
                negative_num += 1
        self._positive_vec = positve_feature_num / positive_num   #类别为1的各个随机向量（特征）的概率分布    the probability of feture num
        self._negative_vec = negative_feature_num / negative_num
        self._p_positive = positive_num / float(len(labels)) #p(y=1)的概率 the probability of positive label
        # return self._positive_vec, self._negative_vec, self._p_positive

    def predict(self, input_data):   #预测函数
        input_vec = self.set_of_word2vec(self.vocab_list, input_data)#测试样本的词条向量
        # np.multiply(self.p1Vect ,inputVec)
        p_positive = self.p_positive   #按照公式需要乘以p(y=1)的值，我们就以此为初始值
        p_negative = (1 - self.p_positive)
        for num in np.multiply(self.positive_vec ,input_vec):  #概率分布和词条向量进行相乘，得出p(x=xi|y=1)的概率，然后相乘
            if (num > 0):
                p_positive *= num
        for num in np.multiply(self.negative_vec ,input_vec):
            if (num > 0):
                p_negative *= num
        print(p_positive, p_negative)
        if (p_positive > p_negative):  #相比，谁大就倾向谁 up to max probability
            return 1
        else:
            return -1

    @property
    def vocab_list(self):
        return self._vocab_list

    @property
    def positive_vec(self):
        return self._positive_vec

    @property
    def negative_vec(self):
        return self._negative_vec

    @property
    def p_positive(self):
        return self._p_positive

if __name__ == "__main__":
    dataSet = [[1, "S"], [1, "M"], [1, "M"], [1, "S"], [1, "S"],
               [2, "S"], [2, "M"], [2, "M"], [2, "L"], [2, "L"],
               [3, "L"], [3, "M"], [3, "M"], [3, "L"], [3, "L"]]
    labels = [-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1]
    bayes = Bayes()
    bayes.train(dataSet, labels)
    print("prediction is:", bayes.predict([2, "S"]))
