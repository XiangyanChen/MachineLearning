#-*-coding:utf-8-*-
import numpy as np
import math
import json
class Node: #结点
    def __init__(self, data = None, lchild = None, rchild = None):
        self.data = data
        self.child = {}

class DecisionTree: #决策树
    def create(self, dataSet, labels):   #ID3算法
        featureSet = self.createFeatureSet(dataSet) #特征集
        def createBranch(dataSet, featureSet):
            label = [row[-1] for row in dataSet]    #按列读取标签
            node = Node()
            #TODO:算法(2)
            if (len(set(label)) == 1):   #说明已经没有需要划分的了
                node.data = label[0]
                node.child = None
                return node
            HD = self.entropy(dataSet)  #数据集的熵
            maxGa = 0   #最大信息增益
            maxGaKey = 0  #最大信息增益的索引
            for key in featureSet: #计算最大信息增益
                gDA = HD - self.conditionalEntropy(dataSet, key, featureSet)  #当前按i维特征划分的信息增益
                gDA = gDA / float(self.entropy(dataSet, key))   #这行注释掉,就是用ID3算法了，否则就是C4.5算法
                print(gDA)
                if (maxGa < gDA):
                    maxGa = gDA
                    maxGaKey = key
            #TODO:算法(4)
            node.data = labels[maxGaKey]
            subFeatureSet = featureSet.copy()
            del subFeatureSet[maxGaKey]#删除特征集（不再作为划分依据）
            for x in featureSet[maxGaKey]:  #这里是计算出信息增益后，知道了需要按哪一维进行划分集合
                subDataSet = [row for row in dataSet if row[maxGaKey] == x] #这个可以得出数据子集
                # print(json.dumps(subDataSet,encoding='UTF-8',ensure_ascii=False))
                node.child[x] = createBranch(subDataSet, subFeatureSet)
            return node
        return createBranch(dataSet, featureSet)
    def classify(self, node, labels, testVec):
        while node != None:
            if (node.data in labels):
                index = labels.index(node.data)
                x = testVec[index]
                for key in node.child:
                    if x == key:
                        node = node.child[key]
            else:
                # print node.data
                break
        return node.data

    def createFeatureSet(self, dataSet):    #创建特征集
        featureSet = {}
        m, n = np.shape(dataSet)
        for i in range(n - 1):  #按列来遍历,n-1代表不存入类别的特征
            column = list(set([row[i] for row in dataSet]))    #按列提取数据
            featureSet[i] = column   #每一行就是每一维的特征值
        return featureSet

    def conditionalEntropy(self, dataSet, i, featureSet): #条件经验熵
        column = [row[i] for row in dataSet]    #读取i列
        entropy = 0
        for x in featureSet[i]:    #划分数据集，并计算
            subDataSet = [row for row in dataSet if row[i] == x]    #按i维的特征划分数据子集
            entropy += (column.count(x) / float(len(column))) * self.entropy(subDataSet)    #按公式来的
        return entropy

    def entropy(self, dataSet, featureKey = -1): #经验熵,默认选最后一列作为特征
        classLabel = [row[featureKey] for row in dataSet]   #读取数据集中的最后一列，也就是类标签
        labelSet = set(classLabel)  #类别的集合
        k = len(labelSet)    #有k个类别，此次数据集中只有2个类别，“是”和“否”
        entropy = 0
        for x in labelSet:  #此为遍历类标签的类别，计算熵
            p = classLabel.count(x) / float(len(classLabel))    #计算概率
            entropy -= p * math.log(p, 2)   #按照公式来的
        return entropy

    def preOrder(self, node, depth = 0):
        if (node != None):
            print(node.data, depth)
            if (node.child != None):
                for key in node.child:
                    print(key)
                    self.preOrder(node.child[key], depth + 1)
    def prune(self, root, dataSet, labels, alpha = 1.0):    #利用生成好的树进行剪枝
        leaveNodeEntropy = {}
        def calcLeaveNum(node): #先计算叶子结点，全局用
            if (node.child == None):    #如果是叶子结点
                return 1
            else:
                sum = 0
                for key in node.child:
                    sum += calcLeaveNum(node.child[key])
                return sum
        leaveNodeNum = calcLeaveNum(root)
        print(leaveNodeNum)
        def saveLeaveNodeEntropy(parent, node): #利用字典来保存叶节点的经验熵
            if (node.child == None):
                index = labels.index(parent.data)   #找出是根据第几特征划分的
                subDataSet = [row for row in dataSet if row[index] == node.data]    #根据数据进行划分
                a = self.entropy(subDataSet)
                leaveNodeEntropy[node] = len(subDataSet) * self.entropy(subDataSet)    #返回第一项和叶子长度
            else:
                for key in node.child:
                    saveLeaveNodeEntropy(node, node.child[key])
        saveLeaveNodeEntropy(None, root)
        print(leaveNodeEntropy)
        def calcLoss(node): #先计算叶子结点，全局用
            if (node.child == None):    #如果是叶子结点
                return 1, leaveNodeEntropy[node]
            else:
                sum = 0
                num = 0
                for key in node.child:
                    leaveNum , firstItem = calcLoss(node.child[key])
                    sum += firstItem
                    num += leaveNum
                return num, sum
        def pruning(parent, node):
            if (node != None and node.child != None):    #说明是非叶子节点
                for key in node.child:
                    pruning(node, node.child[key])
                    if (node.child[key].child == None): #叶子节点才要回缩，然后剪枝叶，否则不理
                        num, firstItem = calcLoss(root)
                        cBefore = firstItem + alpha * num#计算一遍损失函数的值
                        print(calcLeaveNum(root))
                        if (parent != None):
                            for key in parent.child:
                                if (parent.child[key] == node): #要找到当前节点才可以进行节点回缩
                                    parent.child[key] = node.child[key]
                            num, firstItem = calcLoss(root)
                            cAfter = firstItem + alpha * num#计算一遍损失函数的值
                            print(calcLeaveNum(root))
                            if (cBefore<cAfter):    #不剪枝
                                parent.child[key] = node
        pruning(None, root)
if __name__ == "__main__":
    dataSet = [['青年', '否', '否', '一般', '否'],
               ['青年', '否', '否', '好', '否'],
               ['青年', '是', '否', '好', '是'],
               ['青年', '是', '是', '一般', '是'],
               ['青年', '否', '否', '一般', '否'],
               ['中年', '否', '否', '一般', '否'],
               ['中年', '否', '否', '好', '否'],
               ['中年', '是', '是', '好', '是'],
               ['中年', '否', '是', '非常好', '是'],
               ['中年', '否', '是', '非常好', '是'],
               ['老年', '否', '是', '非常好', '是'],
               ['老年', '否', '是', '好', '是'],
               ['老年', '是', '否', '好', '是'],
               ['老年', '是', '否', '非常好', '是'],
               ['老年', '否', '否', '一般', '否']]
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']
    # print dataSet
    tree = DecisionTree()
    node = tree.create(dataSet, labels)
    tree.preOrder(node)
    print(tree.prune(node, dataSet, labels))

    tree.preOrder(node)
    print("-------")
    for line in dataSet:
        print(tree.classify(node, labels, line))
    # tree.createFeatureSet(dataSet)
    # print tree.entropy(dataSet)
    # featureSet = tree.createFeatureSet(dataSet)
    # print tree.conditionalEntropy(dataSet, 0, featureSet)
    # print(json.dumps(tree.createFeatureSet(dataSet),encoding='UTF-8',ensure_ascii=False))
