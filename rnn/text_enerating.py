import nltk
from collections import Counter
import numpy as np

def getVocab(filename='dataset/poetry.txt'):
    with open(filename, encoding='utf-8') as f:
        lines = f.readlines()   #读取每一行
        wordFreqDict = dict()   #将每个词都进行词频计算,取词频高的前多少词用来做词典
        for line in lines:  #遍历每一行
            tokens = dict(nltk.FreqDist(list(line)))    #分词，并且计算词频
            wordFreqDict = dict(Counter(wordFreqDict) + Counter(tokens))    #把每个词的词频相加
        wordFreqTuple = sorted(wordFreqDict.items(), key = lambda x: x[1], reverse=True)    #按词频排序，逆排序
        fw = open('dataset/vocab.txt','w', encoding='utf-8')    #将词典的每个词（这里是每个字作为一个词）写入到文件
        vocab = wordFreqTuple[:8000]    #词典
        for word in vocab:
            fw.write(word[0] + '\n')

def readVocab(filename = 'dataset/vocab.txt'):
    vocab = list()
    fr = open(filename, 'r', encoding='utf-8')
    lines = fr.readlines()
    for line in lines:
        vocab.append(line.rstrip())
    # vocab.remove('\n')
    vocab.remove(':')
    vocab.remove('')
    vocab.remove('')
    vocab.insert(0, 'UNKNOWN')
    vocab.insert(1, 'START')
    vocab.insert(2, 'END')
    return vocab[:6000]

def loadData(filename = 'dataset/poetry.txt'):
    vocab = readVocab() #获取词典，词典是以['词', '典']这样存放的，类似这样
    word2Index = {word: index for index, word in enumerate(vocab)}  #将词典映射成{词:0, 典:1}，类似这样
    fr = open(filename, 'r', encoding='utf-8')  #读取文件，准备数据集
    lines = fr.readlines()  #所有行
    dataSet = list()    #数据集
    labels = list() #标签
    for line in lines:
        poetry = line.split(":")[1].rstrip()  #去除标题
        X = [word2Index.get(word, 0) for word in list(poetry)] #把对应的文字转成索引，形成向量
        y = X.copy()
        X.insert(0, 1)  #1代表是词典里的START，代表开始
        dataSet.append(X)
        y.append(2) #2代表是词典里的END
        labels.append(y)
    return dataSet, labels


class RNN:
    def softmax(self, X):   #softmax函数
        return np.exp(X) / np.sum(np.exp(X))

    def __init__(self):
        self.wordDim = 6000
        wordDim = self.wordDim
        self.hiddenDim = 100
        hiddenDim = self.hiddenDim
        self.Wih = np.random.uniform(-np.sqrt(1. / wordDim), np.sqrt(1. / wordDim), (hiddenDim, wordDim))  #输入层到隐含层的权重矩阵(100, 6000)
        self.Whh = np.random.uniform(-np.sqrt(1. / self.hiddenDim), np.sqrt(1. / self.hiddenDim), (hiddenDim, hiddenDim))  #隐含层到隐含层的权重矩阵(100, 100)
        self.Why = np.random.uniform(-np.sqrt(1. / self.hiddenDim), np.sqrt(1. / self.hiddenDim), (wordDim, hiddenDim))  #隐含层到输出层的权重矩阵(10, 1)

    def train(self,dataSet, labels):    #训练
        N = len(dataSet)
        for i in range(N):
            if (i % 100 == 0 and i >= 100):
                self.calcEAll(dataSet[i-100:i], labels[i-100:i])
            self.backPropagation(dataSet[i], labels[i])

    def forward(self, data):  #前向传播，原则上传入一个数据样本和标签
        T = len(data)
        output = np.zeros((T, self.wordDim, 1)) #输出
        hidden = np.zeros((T+1, self.hiddenDim, 1)) #隐层状态
        for t in range(T): #时间循环
            X = np.zeros((self.wordDim, 1)) #构建(6000,1)的向量
            X[data[t]][0] = 1   #将对应的值置为1，形成词向量
            Zh = np.dot(self.Wih, X) + np.dot(self.Whh, hidden[t-1])   #(100, 1)
            ah = np.tanh(Zh)   #(100, 1)，隐层值
            hidden[t] = ah
            Zk = np.dot(self.Why, ah)   #(6000,1)
            ak = self.softmax(Zk)   #(6000, 1)，输出值
            output[t] = ak #把index写进去
        return hidden, output

    def backPropagation(self, data, label, alpha = 0.0002):  #反向传播
        hidden, output = self.forward(data)  #(N, 6000)
        T = len(output) #时间长度=词向量的长度
        deltaHPre = np.zeros((self.hiddenDim, 1))   #前一时刻的隐含层偏导
        WihUpdata = np.zeros(self.Wih.shape)    #权重更新值
        WhhUpdata = np.zeros(self.Whh.shape)
        WhyUpdata = np.zeros(self.Why.shape)
        for t in range(T-1, -1, -1):
            X = np.zeros((self.wordDim, 1))  # (6000,1)
            X[data[t]][0] = 1   #构建出词向量
            output[t][label[t]][0] -= 1 #求导后，输出结点的误差跟output只差在i=j时需要把值减去1
            deltaK = output[t].copy()   #输出结点的误差
            deltaH = np.multiply(np.add(np.dot(self.Whh.T, deltaHPre),np.dot(self.Why.T, deltaK)), (1 - (hidden[t] ** 2)))  #隐含层结点误差
            deltaHPre=deltaH.copy()
            WihUpdata += np.dot(deltaH, X.T)
            WhhUpdata += np.dot(deltaH, hidden[t-1].T)
            WhyUpdata += np.dot(deltaK, hidden[t].T)
        self.Wih -= alpha * WihUpdata
        self.Whh -= alpha * WhhUpdata
        self.Why -= alpha * WhyUpdata

    def calcSingleE(self, output, label):
        N = len(output)
        sum = 0
        for i in range(N):
            try:
                sum -= np.log(output[i][label[i]][0])
            except IndexError as e:
                print(np.shape(output))
                print(np.shape(label))
        # print(sum / N)
        return sum / N

    def calcEAll(self, dataSet, labels):
        N = len(dataSet)
        sum = 0
        for i in range(N):
            hidden, output = self.forward(dataSet[i])
            sum += self.calcSingleE(output, labels[i])
        print(sum / N)

if __name__ == '__main__':
    dataSet, labels = loadData()
    rnn = RNN()
    rnn.train(dataSet, labels)
