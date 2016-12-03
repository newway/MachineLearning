# -*- encoding: utf-8 -*-

from numpy import *

def loadSimpleData():
    datMat = matrix([[1. , 2.1],
                     [2. , 1.1],
                     [1.3, 1. ],
                     [1. , 1. ],
                     [2. , 1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels

#单层决策分类函数
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = ones((shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray

#构建单层决策树
def buildStump(dataArr, classLabels, D):
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T
    m,n = shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClasEst = mat(zeros((m, 1)))
    minError = inf
    for i in range(n):
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin)/numSteps
        for j in range(-1, int(numSteps)+1):
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                predictVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = mat(ones((m, 1)))
                errArr[predictVals==labelMat] = 0
                weightedError = D.T * errArr
                #print "split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % \
                #      (i, threshVal, inequal, weightedError)
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst

#基于单层决策树的AdaBoost训练函数
def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    weakClassArr = []
    m = shape(dataArr)[0]
    #D 为数据点的权重（概率分布）， 对于某个数据点，某次分类出错，则提高此数据点的权重，便于下一轮单层分类更有针对性的进行处理
    D = mat(ones((m, 1))/m)
    aggClassEst = mat(zeros((m, 1)))
    for i in range(numIt):
        #单层决策分类，可以使用其他分类方法
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        print "D: ", D.T
        #本次分类结果在总分类结果中的权重， 正确率越高权重越大，对最终分类结果影响越大
        alpha = float(0.5*log((1.0-error)/max(error, 1e-16)))
        print "alpha: ", alpha
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print "classEst: ", classEst.T
        #根据出错率改变数据点的权重
        expon = multiply(-1*alpha*mat(classLabels).T, classEst)
        print "expon: ", expon
        D = multiply(D, exp(expon))
        D = D/D.sum()
        #将本次分类的结果乘以权重计入最终分类结果
        aggClassEst += alpha*classEst
        print "aggClassEst: ", aggClassEst.T
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))
        print "aggErrors: ", aggErrors
        errorRate = aggErrors.sum()/m
        print "total error: ", errorRate, "\n"
        if errorRate == 0.0:
            break
    return weakClassArr

def adaClassify(datToClass, classifierArr):
    dataMatrix = mat(datToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        #'alpha' 记录了单层分类器结果对总结果的权重
        aggClassEst += classifierArr[i]['alpha'] * classEst
        print aggClassEst
    return sign(aggClassEst)

if __name__ == '__main__':
    datMat, classLabels = loadSimpleData()
    print datMat, classLabels
    D = mat(ones((5,1))/5)
    buildStump(datMat, classLabels, D)
    classifierArr = adaBoostTrainDS(datMat, classLabels, 30)
    print classifierArr
    final_result = adaClassify([0, 0], classifierArr)
    print final_result