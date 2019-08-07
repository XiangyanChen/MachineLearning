"""
This is the implementation of Logistic Regression,
which is accessible in https://github.com/FlameCharmander/MachineLearning,
accomplished by FlameCharmander,
and my csdn blog is https://blog.csdn.net/tudaodiaozhale,
contact me via 13030880@qq.com.
"""
import numpy as np
from sklearn import svm
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
        maxIter = 10 #最大循环次数
        iter = 0
        N = len(dataSet)   #数据的行数
        M = len(dataSet[0])    #数据的列数，维数
        alpha = np.zeros(N)
        while iter < maxIter:
            if (iter % 50 == 0):
                print(iter)
            iter += 1
            flag = False
            for i in range(N):  #外循环
                alpha1_old = alpha[i].copy()   #未更新的alpha,也就是alpha_old
                y1 = labels[i]  #y1
                x1 = dataSet[i]
                g1 = self.calcG(alpha, labels, x1, dataSet, b)
                alpha_index1 = -1    #存储不满足KKT条件的alpha_index1
                if alpha1_old == 0 and y1 * g1 < 1:   #判断是否满足KKT条件 (7.111)
                    alpha_index1 = i
                if alpha1_old > 0 and alpha[i] < C and y1 * g1 != 1:   #(7.112)
                    alpha_index1 = i
                if alpha1_old == C and y1 * g1 > 1:   #(7.1132)
                    alpha_index1 = i
                if alpha_index1 == -1:   #说明满足KKT条件，继续下一次循环来找alpha1
                    continue
                E1 = g1 - y1    #(7.105)
                alpha_index2 = -1
                # max_E1_E2 = -1
                # E2 = 0
                if E1 > 0: #正的话要找E2的最小值，反之同理
                    selectedE2 = np.inf
                else:
                    selectedE2 = -np.inf
                for j in range(N): #内循环
                    if i != j:  #相等就没法选了
                        yj = labels[j]
                        gj = self.calcG(alpha, labels, dataSet[j], dataSet, b)
                        Ej = gj - yj
                        # if (np.abs(Ej + E1) > np.abs(E2 + E1)):
                        #     alpha_index2 = j
                        #     E2 = Ej
                        if E1 > 0:  #说明要选最小的E2
                            if Ej < selectedE2:
                                selectedE2 = Ej
                                alpha_index2 = j
                        else:
                            if Ej > selectedE2:
                                selectedE2 = Ej
                                alpha_index2 = j
                if (alpha_index2 == -1):
                    continue
                '''
                   此时应该选到了alpha2了
                '''
                L = 0  # P126末尾两段
                H = C
                E2 = selectedE2
                y2 = labels[alpha_index2]
                alpha2_old = alpha[alpha_index2].copy()
                x2 = dataSet[alpha_index2]
                if (y1 != y2):  #alpha2取值范围必须限制在L<alpha2<H
                    L = np.maximum(0, alpha2_old - alpha1_old)  # L
                    H = np.minimum(C, C + alpha2_old - alpha1_old)  #H
                else:
                    L = np.maximum(0, alpha2_old + alpha1_old - C)  # L
                    H = np.minimum(C, alpha2_old + alpha1_old)    #H
                eta = self.calcK(x1, x1) + self.calcK(x2, x2) - 2 * self.calcK(x1, x2)
                if eta == 0:    #没法选
                    continue
                alpha2_new_unc = alpha2_old + (y2 * (E1 - E2)) / eta
                if (alpha2_new_unc > H):  # (7.108)
                    alpha2_new = H
                elif (alpha2_new_unc < L):
                    alpha2_new = L
                else:
                    alpha2_new = alpha2_new_unc
                alpha1_new = alpha1_old + y1 * y2 * (alpha2_old - alpha2_new)   #(7.109)

                b1new = -E1 - y1 * self.calcK(x1, x1) * (alpha1_new - alpha1_old) - y2 * self.calcK(x2, x1) * (alpha2_new - alpha2_old) + b #(7.115)
                b2new = -E2 - y1 * self.calcK(x1, x2) * (alpha1_new - alpha1_old) - y2 * self.calcK(x2, x2) * (alpha2_new - alpha2_old) + b #(7.116)

                if (alpha1_new > 0 and alpha1_new < C and alpha2_new > 0 and alpha2_new < C):
                    b = b1new
                else:
                    b = (b1new + b2new) / 2
                alpha[alpha_index1] = alpha1_new
                alpha[alpha_index2] = alpha2_new
                # break
            self.weights = np.dot(np.multiply(alpha, labels), dataSet)   #权重
            self.b = b
            # print(alpha)
        return self.weights, self.b

    def predict(self, x):
        return sign(np.dot(self.weights, x) + self.b)

    def calcK(self, x1, x2):  #线性核函数，返回内积
        return np.dot(x1, x2)

    def calcG(self, alpha, labels, data, dataSet, b):   #计算g
        sum = 0
        for j in range(len(alpha)):
            sum += alpha[j] * labels[j] * self.calcK(data, dataSet[j]) #g(x)的计算
        return sum + b



if __name__ == '__main__':
    clf = svm.SVC(kernel='linear', C=1.0)  # class


    dataSet, labels = loadDataSet()
    clf.fit(dataSet, labels)  # training the svc model
    w = clf.coef_[0]
    print("w = ", end='')
    print(w)
    print("b = ", end='')
    print(clf.intercept_)


    svm = SVM()
    weights, b = svm.train(dataSet, labels)
    # weights, b = svm.train(dataSet[:80], labels[:80])
    print(weights, b)
    x = [1, 2]
    for i, x in enumerate(dataSet[:80]):
        result = svm.predict(x)
        if (int(labels[i]) == result):
            print(True)
        else:
            print(False)
        # print(result)
