import numpy as np

def loadDataSet():  #加载文件
    data = list()
    labels = list()
    with open('data/testSet.txt') as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip().split('\t')
            data.append([float(line[0]), float(line[1])])
            labels.append(float(line[-1]))
    return data, labels

def sign(x):    #符号函数
    if x >= 0:
        return 1
    else:
        return -1

class SVM:
    def train(self, dataSet, labels):   #训练并返回权重和偏置
        b = 0   #偏置
        C = 1 #惩罚系数
        flag = True    #检验是否全部都满足KKT条件
        maxIter = 100 #最大循环次数
        iter = 0
        N = len(dataSet)   #数据的行数
        M = len(dataSet[0])    #数据的列数，维数
        alpha = np.zeros(N)
        while iter < maxIter:
            print(iter)
            iter += 1
            flag = False
            for i in range(N):  #外循环
                alpha1old = alpha[i].copy()   #未更新的alpha,也就是alpha_old
                y1 = labels[i]  #y1
                data1 = dataSet[i]
                g1 = self.calcG(alpha, labels, data1, dataSet, b)
                alphaIndex1 = -1    #存储不满足KKT条件的alphaIndex1
                if alpha1old == 0:   #判断是否满足KKT条件 (7.111)
                    if y1 * g1 < 1:
                        alphaIndex1 = i
                        flag = True
                if alpha1old > 0 and alpha[i] < C:   #(7.112)
                    if y1 * g1 != 1:
                        alphaIndex1 = i
                        flag = True
                if alpha1old == C:   #(7.1132)
                    if y1 * g1 <= 1:
                        alphaIndex1 = i
                        flag = True
                if alphaIndex1 == -1:   #说明满足KKT条件，继续下一次循环来找alpha1
                    continue
                E1 = g1 - y1    #(7.105)
                alphaIndex2 = -1
                if E1 > 0: #正的话要找E2的最小值，反之同理
                    selectedE2 = np.inf
                else:
                    selectedE2 = -np.inf
                for j in range(N): #内循环
                    if i != j:  #相等就没法选了
                        yj = labels[j]
                        gj = self.calcG(alpha, labels, dataSet[j], dataSet, b)
                        Ej = gj - yj
                        if E1 > 0:  #说明要选最小的E2
                            if Ej < selectedE2:
                                selectedE2 = Ej
                                alphaIndex2 = j
                        else:
                            if Ej > selectedE2:
                                selectedE2 = Ej
                                alphaIndex2 = j
                '''
                   此时应该选到了alpha2了
               '''
                L = 0  # P126末尾两段
                H = 0
                y2 = labels[alphaIndex2]
                alpha2old = alpha[alphaIndex2].copy()
                data2 = dataSet[alphaIndex2]
                E2 = selectedE2
                if (y1 == y2):  #alpha2取值范围必须限制在L<alpha2<H
                    L = np.maximum(0, alpha2old - alpha1old)  # L
                    H = np.minimum(C, C + alpha2old - alpha1old)  #H
                else:
                    L = np.maximum(0, alpha2old + alpha1old - C)  # L
                    H = np.minimum(C, C + alpha2old + alpha1old)    #H
                eta = self.calcK(data1, data1) + self.calcK(data2, data2) - 2 * self.calcK(data1, data2)
                if eta == 0:    #没法选
                    continue
                alpha2new = alpha2old + (y2 * (E1 - E2)) / eta
                if (alpha2new > H):  # (7.108)
                    alpha[alphaIndex2] = H
                elif (alpha2new < L):
                    alpha[alphaIndex2] = L
                else:
                    alpha[alphaIndex2] = alpha2new
                alpha1new = alpha1old * y1 * y2 * (alpha2old - alpha2new)   #(7.109)
                alpha[alphaIndex1] = alpha1new
                b1new = -E1 - y1 * self.calcK(data1, data1) * (alpha1new - alpha1old) - y2 * self.calcK(data2, data1) * (alpha2new - alpha2old) + b #(7.115)
                b2new = -E2 - y1 * self.calcK(data1, data1) * (alpha1new - alpha1old) - y2 * self.calcK(data2, data2) * (alpha2new - alpha2old) + b #(7.116)
                if (alpha1new > 0 and alpha1new < C):
                    b = b1new
                else:
                    b = (b1new + b2new) / 2
        print(alpha)
        weights = np.dot(np.multiply(alpha, labels), dataSet)   #权重
        return weights, b

    def calcK(self, data1, data2):  #线性核函数，返回内积
        return np.dot(data1, data2)

    def calcG(self, alpha, labels, data, dataSet, b):   #计算g
        sum = 0
        for j in range(len(alpha)):
            sum += alpha[j] * labels[j] * self.calcK(data, dataSet[j]) #g(x)的计算
        return sum + b

if __name__ == '__main__':
    dataSet, labels = loadDataSet()
    svm = SVM()
    weights, b = svm.train(dataSet, labels)
    print(weights, b)
    x = [1, 2]
    f = sign(np.dot(weights, x) + b)
    print(f)
