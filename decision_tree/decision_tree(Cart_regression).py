# --*-- coding:utf-8 --*--
import numpy as np

class RegressionTree:   #回归树
    def loadDataSet(self, fileName):    #加载数据
        dataMat = []
        fr = open(fileName)
        for line in fr.readlines(): #遍历每一行
            curLine = line.strip().split('\t')
            fltLine = list(map(float, curLine))   #将里面的值映射成float,否则是字符串类型的
            dataMat.append(fltLine)
        return dataMat

    def binSplitDataSet(self, dataSet, feature, value): #按某列的特征值来划分数据集
        mat0 = dataSet[np.nonzero(dataSet[:, feature] > value)[0], :]   #勘误：这里跟书上不一样，需修改
        mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0], :]  #np.nonzero(...)[0]返回一个列表
        return mat0, mat1
    def regLeaf(self, dataSet): #将均值作为叶子节点
        return np.mean(dataSet[:, -1])

    def regErr(self, dataSet):#计算误差
        return np.var(dataSet[:, -1]) * np.shape(dataSet)[0]    #方差乘以行数

    def createTree(self, dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
        feat, val = self.chooseBestSplit(dataSet, leafType, errType, ops)
        if feat == None:    return val  #说明是叶节点，直接返回均值
        retTree = {}
        retTree['spInd'] = feat #记录是用哪个特征作为划分
        retTree['spVal'] = val  #记录是用哪个特征作为划分（以便于查找的时候，相等进入左树，不等进入右树）
        lSet, rSet = self.binSplitDataSet(dataSet, feat, val)   #按返回的特征来选择划分子集
        retTree['left'] = self.createTree(lSet, leafType, errType, ops) #用划分的2个子集的左子集，递归建树
        retTree['right'] = self.createTree(rSet, leafType, errType, ops)
        return retTree

    def chooseBestSplit(self, dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
        tolS = ops[0] #容许的误差下降值
        tolN = ops[1]   #划分的最少样本数
        if len(set(dataSet[:, -1].T.tolist()[0])) == 1: #类标签的值都是一样的，说明没必要划分了，直接返回
            return None, leafType(self, dataSet)
        m, n = np.shape(dataSet)    #m是行数，n是列数
        S = errType(self, dataSet)    #计算总体误差
        bestS = np.inf  #np.inf是无穷大的意思，因为我们要找出最小的误差值，如果将这个值设得太小，遍历时很容易会将这个值当成最小的误差值了
        bestIndex = 0
        bestValue = 0
        for featIndex in range(n-1):    #遍历每一个维度
            for splitVal in set(dataSet[:,featIndex].T.A.tolist()[0]): #选出不同的特征值，进行划分,勘误：这里跟书上不一样，需修改
                mat0, mat1 = self.binSplitDataSet(dataSet, featIndex, splitVal) #子集的划分
                if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):    #划分的两个数据子集，只要有一个小于4，就说明没必要划分
                    continue
                newS = errType(self, mat0) + errType(self, mat1)    #计算误差
                if newS < bestS:    #更新最小误差值
                    bestIndex = featIndex
                    bestValue = splitVal
                    bestS = newS
        if (S - bestS) < tolS:  #检查新切分能否降低误差
            return None, leafType(self, dataSet)
        mat0, mat1 = self.binSplitDataSet(dataSet, bestIndex, bestValue)
        if (np.shape(mat0)[0] < tolN) or(np.shape(mat1)[0] < tolN): #检查是否需要划分（如果两个子集的任一方小于4则没必要划分）
            return None, leafType(self, dataSet)
        return bestIndex, bestValue

if __name__ == '__main__':
    regTree = RegressionTree()
    myMat = regTree.loadDataSet('ex0.txt')
    myMat = np.mat(myMat)
    print(regTree.createTree(myMat))
    # print myMat[:, 1]
    # regTree.binSplitDataSet(np.mat(np.eye(4)), 1, 0.5)
    # print myMat[[1, 2], :]
    # print myMat
    # print np.var(myMat[:, -1]) * np.shape(myMat)[0]

    print(myMat[:,1].T.A.tolist()[0])
