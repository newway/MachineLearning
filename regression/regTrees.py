# -*- encoding: utf-8 -*-

from numpy import *

def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float, curLine)
        dataMat.append(fltLine)
    return dataMat

def binSplitDataSet(dataSet, feature, value):
    #满足条件的点的第一维坐标值数组
    mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :][0]
    mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0],:][0]
    print mat0
    return mat0, mat1

def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None:
        return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree

def regLeaf(dataSet):
    return mean(dataSet[:, -1])

def regErr(dataSet):
    return var(dataSet[:, -1]) * shape(dataSet)[0]

def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    tolS = ops[0]
    tolN = ops[1]
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    m,n = shape(dataSet)
    S = errType(dataSet)
    bestS = inf
    bestIndex = 0
    bestValue = 0
    for featIndex in range(n-1):
        for splitVal in set(dataSet[:, featIndex]):
            mat0,mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            #如果划分出来的数据集行数小于阈值则停止划分, 防止过拟合
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
                continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestS = newS
                bestIndex = featIndex
                bestValue = splitVal
    #如果误差减少不大则退出, 防止过拟合
    if (S - bestS) < tolS:
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    #如果划分出来的数据集很小则退出
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    return bestIndex, bestValue

def isTree(obj):
    return (type(obj).__name__ == 'dict')

def getMean(tree):
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    #为什么求平均是左右子树除2？？？，左右子树包含的节点数可能不一样多
    return (tree['left'] + tree['right']) / 2.0

#回归树剪枝函数：叶节点合并检查，如果误差减小则合并
def prune(tree, testData):
    if shape(testData)[0] == 0:
        return getMean(tree)
    if (isTree(tree['right'])) or isTree(tree['left']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)
    #当前是叶子节点父节点，计算测试数据划分效果
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(power(lSet[:,-1] - tree['left'], 2)) + sum(power(rSet[:,-1] - tree['right'], 2))
        treeMean = (tree['left'] + tree['right']) / 2.0
        errorMerge = sum(power(testData[:,-1] - treeMean, 2))
        if errorMerge < errorNoMerge:
            print "merging"
            return treeMean
        else:
            return tree
    else:
        return tree

#模型树的叶节点生成函数

#执行简单的线性回归，获取回归系数
def linearSolve(dataSet):
    m,n = shape(dataSet)
    X = mat(ones((m, n)))
    Y = mat(ones((m, n)))
    X[:, 1:n] = dataSet[:, 0:n-1]
    Y = dataSet[:,-1]
    xTx = X.T * X
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse, \n\
                        try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws, X, Y
#当数据不再需要切分时，返回叶节点的模型，回归系数
def modelLeaf(dataSet):
    ws,X,Y = linearSolve(dataSet)
    return ws

#计算误差
def modelErr(dataSet):
    ws,X,Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat), 2)

def regTreeEval(model, inDat):
    return float(model)

def modelTreeEval(model, inDat):
    n = shape(inDat)[1]
    X = mat(ones((1, n+1)))
    X[:, 1:n+1] = inDat
    return float(X*model)

#利用树对输入数据进行分类预测
def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree):
        return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)

def createForeCast(tree, testData, modelEval=regTreeEval):
    m = len(testData)
    yHat = mat(zeros((m, 1)))
    for i in range(m):
        yHat[i,0] = treeForeCast(tree, mat(testData[i]), modelEval)
    return yHat

