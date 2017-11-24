# -*- encoding: utf-8 -*-
import random


#有放回采样, 采样数据量等同原始数据集，初始训练集中约有63.2%的样本出现在采样集中
#剩下约36.8%的样本可用作验证集来对泛化性能进行“包外估计”
def sampleData(data):
    output = []
    num = len(data)
    for i in num:
        r = random.randint(0, m-1)
        output.append(data[r])
    return output

#重复采样T个样本集，训练出T个基学习器
#Bagging主要关注降低方差(增加稳定性)，因此在不减枝决策树、神经网络等易受样本扰动的学习器上效果更明显
def bagging(data, T):
    bags = []
    for i in T:
        dataSet = sampleData(data)
        basic = buildBasicLearner(dataSet)
        bags.append(basic)
    return bags

#多个学习器进行表决，假设分类为1、-1
def generateResult(bags, test):
    res = 0
    for item in bags:
        res += item(test)
    return res

#random forest
# use sampled datat and features
# 假设数据有d个特征，一般随机选择log2 d个特征，然后从属性子集中选择1个最优属性进行划分
#与bagging相比，除了数据样本扰动，还有属性扰动，增加了基学习器的多样性，集成之后增加了泛化性能，且一般比bagging训练效率高
def randomForest(data, T):
    forest = []
    for i in T:
        dataSet = sampleData(data)
        tree = createTree(dataSet[:[random feature]], labels)
        forest.append(tree)
    return forest

#ensemble
E = ⋶ - A
基学习器误差⋶越小，多样性A越大，则最终的泛化误差越小
