# --*-- coding:utf-8 --*--
import numpy as np

class Node: #结点
    def __init__(self, data = None, lchild = None, rchild = None):
        self.data = data
        self.child = {} #需要用字典的key来做边的值('是'，'否')

class DecisionTree4Cart:    #分类与回归树
    def create(self, dataSet, labels):
        featureSet = self.createFeatureSet(dataSet)
        def createBranch(dataSet, featureSet):
            classLabel = [row[-1] for row in dataSet]    #按列读取标签
            node = Node()
            if (len(set(classLabel)) == 1):   #说明已经没有需要划分的了
                node.data = classLabel[0]
                node.child = None
                return node
            minGini = 1.1    #    #不会超过1
            minGiniIndex = -1
            minGiniFeature = None
            for key in featureSet:
                feature = featureSet[key]
                for x in feature:
                    gini = self.calcConditionalGini(dataSet, key, x) #计算基尼指数
                    print(gini)
                    if (minGini > gini):    #比较得出最小的基尼指数
                        minGini = gini
                        minGiniIndex = key
                        minGiniFeature = x
            node.data = labels[minGiniIndex]
            subFeatureSet = featureSet.copy()
            del subFeatureSet[minGiniIndex]#删除特征集（不再作为划分依据）
            subDataSet1 = [row for row in dataSet if row[minGiniIndex] == minGiniFeature]
            node.child[minGiniFeature] = createBranch(subDataSet1, subFeatureSet)
            subDataSet2 = [row for row in dataSet if row[minGiniIndex] != minGiniFeature]
            node.child["other"] = createBranch(subDataSet2, subFeatureSet)
            return node
        return createBranch(dataSet, featureSet)

    def calcConditionalGini(self, dataSet, featureIndex, value):   #计算基尼指数
        conditionalGini = 0
        """
        可以看出下面的代码使按公式5.25来的吗？
        """
        subDataSet1 = [row for row in dataSet if row[featureIndex] == value]    #按值划分数据集，这是第一个数据集
        conditionalGini += len(subDataSet1) / float(len(dataSet)) * self.calcGini(subDataSet1)
        subDataSet2 = [row for row in dataSet if row[featureIndex] != value]    #第二个数据集
        conditionalGini += len(subDataSet2) / float(len(dataSet)) * self.calcGini(subDataSet2)
        return conditionalGini

    def calcGini(self, dataSet, featureKey = -1):   #计算基尼指数
        classLabel = [row[featureKey] for row in dataSet]
        labelSet = set(classLabel)  #类别的集合
        gini = 1
        for x in labelSet:  #此为遍历类标签的类别，计算熵
            gini -= ((classLabel.count(x) / float(len(dataSet))) ** 2) #按公式5.24来
        return gini

    def preOrder(self, node, depth = 0):    #先序遍历
        if (node != None):
            print(node.data, depth)
            if (node.child != None):
                for key in node.child:
                    print(key)
                    self.preOrder(node.child[key], depth + 1)

    def createFeatureSet(self, dataSet):    #创建特征集
        featureSet = {}
        m, n = np.shape(dataSet)
        for i in range(n - 1):  #按列来遍历,n-1代表不存入类别的特征
            column = list(set([row[i] for row in dataSet]))    #按列提取数据
            featureSet[i] = column   #每一行就是每一维的特征值
        return featureSet

    def classify(self, node, labels, testVec):  #类别判断
        while node != None:
            if (node.data in labels):   #用来判断是否内部结点，内部结点就继续往下找
                index = labels.index(node.data) #非根结点意味着是根据某个特征划分的，找出该特征的索引
                x = testVec[index]
                for key in node.child:  #遍历结点孩子字典，用key来做权值来判断该往左结点移动还是右节点
                    if x == key:
                        node = node.child[key]
                        break
                else:
                    node = node.child['other']
            else:
                break
        return node.data
if __name__ == '__main__':
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
    tree = DecisionTree4Cart()
    node = tree.create(dataSet, labels)
    tree.preOrder(node)
    for dataLine in dataSet:
        print(tree.classify(node, labels, dataLine))
