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
