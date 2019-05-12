#-*-coding:utf-8-*-
import numpy as np
import math
class Node: #结点
    def __init__(self, data = None, lchild = None, rchild = None):
        self.data = data
        self.child = {}

class DecisionTree: #决策树
    def create(self, dataSet, labels, option="ID3"):
        """
        :param dataSet:
        :param labels:
        :param option: "(ID3|C4.5)"
        :return:
        """
        feature_set = self.create_feature_set(dataSet) #特征集
        def create_branch(dataSet, feature_set):
            label = [row[-1] for row in dataSet]    #按列读取标签
            node = Node()
            # TODO:算法(2)
            if (len(set(label)) == 1):   #this means this dataset needn't split
                node.data = label[0]
                node.child = None
                return node
            HD = self.entropy(dataSet)  #数据集的熵
            max_ga = 0   #最大信息增益
            max_ga_key = 0  #最大信息增益的索引
            for key in feature_set: #计算最大信息增益
                g_DA = HD - self.conditional_entropy(dataSet, key, feature_set)  #当前按i维特征划分的信息增益
                if option=="C4.5":
                    g_DA = g_DA / float(self.entropy(dataSet, key))   #这行注释掉,就是用ID3算法了，否则就是C4.5算法
                if (max_ga < g_DA):
                    max_ga = g_DA
                    max_ga_key = key
            #TODO:算法(4)
            node.data = labels[max_ga_key]
            sub_feature_set = feature_set.copy()
            del sub_feature_set[max_ga_key]#删除特征集（不再作为划分依据）
            for x in feature_set[max_ga_key]:  #这里是计算出信息增益后，知道了需要按哪一维进行划分集合
                sub_data_set = [row for row in dataSet if row[max_ga_key] == x] #这个可以得出数据子集
                node.child[x] = create_branch(sub_data_set, sub_feature_set)    #continue to split the sub data set
            return node
        return create_branch(dataSet, feature_set)

    def classify(self, node, description, sample):
        """
        :param node: node means tree node,
        :param description: description means feature description,
        :param sample: sample means test sample
        :return: classified result
        """
        while node != None:
            if (node.data in description):  #if node.data doesn't exist in description means this node is a leaf node
                index = description.index(node.data) #
                x = sample[index]
                for key in node.child:  #iterate all child
                    if x == key:
                        node = node.child[key]
            else:
                break
        return node.data

    def create_feature_set(self, dataSet):    #创建特征集,特征集是样本每个维度的值域（取值）
        feature_set = {}
        m, n = np.shape(dataSet)    #m means rows, n means columns
        for axis in range(n - 1):  #按列来遍历,n-1代表不存入类别的特征(标签不存入)
            column = list(set([row[axis] for row in dataSet]))    #按列提取数据，并用set过滤
            feature_set[axis] = column   #每一行就是每一维的特征值
        return feature_set

    def conditional_entropy(self, dataSet, feature_key, feature_set): #条件经验熵
        value = [row[feature_key] for row in dataSet]    #读取i列
        entropy = 0
        for x in feature_set[feature_key]:    #划分数据集，并计算
            sub_data_set = [row for row in dataSet if row[feature_key] == x]    #按i轴的特征划分数据子集
            entropy += (value.count(x) / float(len(value))) * self.entropy(sub_data_set)    #p*entropy(sub_data_set)
        return entropy

    def entropy(self, dataSet, feature_key = -1): #计算熵
        """
        :param dataSet:
        :param feature_key: we calculate entropy based on feature key
        :return:
        """
        value_list = [row[feature_key] for row in dataSet] #get the value by axis
        feature_set = set(value_list)  #range of this axis
        entropy = 0
        for x in feature_set:  #iterate every
            p = value_list.count(x) / float(len(value_list))
            entropy -= p * math.log(p, 2)   #entropy=-p(log(p))
        return entropy

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
    description = ['年龄', '有工作', '有自己的房子', '信贷情况'] #the description of samples, the internal node data is one of this description
    tree = DecisionTree()
    node = tree.create(dataSet, description, option="C4.5")
    # validate
    for line in dataSet:
        print(tree.classify(node, description, line))
