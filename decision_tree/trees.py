# -*- encoding: utf-8 -*-

from math import log

def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1

    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob, 2)

    return shannonEnt

def createDataSet():
    dataSet = [[1,1,'yes'],
               [1,1,'yes'],
               [1,0,'no'],
               [0,1,'no'],
               [0,1,'no']]
    labels = ['no surfacing', 'flilppers']
    return dataSet, labels

def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)

    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):
        #属性i上的所有特征值
        featList = [item[i] for item in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            #返回属性i的值为value的所有数据，返回数据不包含属性i
            subDataSet = splitDataSet(dataSet, i, value)
            #属性i值为value的概率
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        #newEntropy 为当前属性所有特征值进行划分的所有子集的熵之和
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
            #print bestInfoGain,bestFeature
    return bestFeature

def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount, key=lambda x:x[vote], reverse=True)
    return sortedClassCount[0][0]

#labels are feature names, not class list
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueValues = set(featValues)
    for value in uniqueValues:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

#inputTree:已构造好的决策树, featLabels:所有特征值名字, testVec:待分类数据
def classify(inputTree, featLabels, testVec):
    #找到树根的决策特征
    firstStr = inputTree.keys()[0]
    #不同特征值对应子节点的dict
    secondDict = inputTree[firstStr]
    #树根决策特征在数据特征集中的下标
    featIndex = featLabels.index(firstStr)
    #找到待测试数据在决策特征上的值key对应的子树进行递归查找
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            #如果决策进入的分支是子树
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            #进入叶子节点,确定分类结果
            else:
                classLabel = secondDict[key]
    return classLabel

if __name__ == '__main__':
    myDat,labels = createDataSet()
    print myDat
    print labels
    dupLabels = labels[:]
    myTree = createTree(myDat, dupLabels)
    print myTree, labels
    print classify(myTree, labels, [1, 0])
    print classify(myTree, labels, [1, 1])
