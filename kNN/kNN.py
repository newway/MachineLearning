# -*- encoding:utf-8 -*-
from numpy import *
import operator

class kNN():
    def __init__(self, k, characternumber):
        #make a sample dataset for demo
        self.k = k
        self.characternumber = characternumber

    def createDataSet(self):
        group = array([[1.0, 1.1], [1.0, 1.0], [0,0], [0, 0.1],[0.1,0.1]])
        labels = ['A', 'A', 'B', 'B', 'B']
        return group, labels

    #各个特征值使用相同的权重,可改进为不同权重
    def classify0(self, inX, dataSet, labels, k):
        dataSetSize = dataSet.shape[0]
        #dataSet每行数据到inX的距离， inX为行向量按照dataSetSize扩展
        diffMat = tile(inX, (dataSetSize,1)) - dataSet
        sqDiffMat = diffMat ** 2
        #axis=1 按行求和
        sqDistances = sqDiffMat.sum(axis=1)
        distances = sqDistances ** 0.5
        sortedDistIndicies = distances.argsort()
        classCount = {}
        for i in range(k):
            ithVoteclass = labels[sortedDistIndicies[i]]
            classCount[ithVoteclass] = classCount.get(ithVoteclass, 0) + 1

        #sortedClassCount = sorted(classCount.iteritems(), key=lambda x:x[1], reverse=True)
        sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
        #print sortedClassCount[0][0]
        return sortedClassCount[0][0]

    def file2matrix(self, filename):
        with open(filename) as fr:
            arraylines = fr.readlines()
            numberoflines = len(arraylines)
            returnmatrix = zeros((numberoflines, self.characternumber))
            classLabels = []
            index = 0
            for line in arraylines:
                #去除行结束符
                line = line.strip()
                listFromLine = line.split('\t')
                returnmatrix[index,:] = listFromLine[0:3]
                classLabels.append(int(listFromLine[-1]))
                #classLabels.append(listFromLine[-1])
                index += 1
        return returnmatrix,classLabels

    def autoNorm(self, dataSet):
        #min(0) 按列求最值
        minVals = dataSet.min(0)
        maxVals = dataSet.max(0)
        ranges = maxVals - minVals
        normalDataSet = zeros(shape(dataSet))
        m = dataSet.shape[0]
        normalDataSet = dataSet - tile(minVals,(m,1))
        normalDataSet = normalDataSet/tile(ranges,(m,1))
        return normalDataSet, ranges, minVals

    #测试分类算法的效果,错误率
    def datingClassTest(self):
        #选取测试样本的比例
        hoRatio = 0.50
        datingDataMat, datingLables = self.file2matrix('testDating.txt')
        normalMat, ranges, minvals = self.autoNorm(datingDataMat)
        m = normalMat.shape[0]
        numTestVecs = int(m*hoRatio)
        errorCount = 0.0
        for i in range(numTestVecs):
            classfierResult = self.classify0(normalMat[i,:], normalMat[numTestVecs:m,:], datingLables[numTestVecs:m], self.k)
            print '%dth test result is %d, and labels is %d' % (i+1, classfierResult, datingLables[i])
            if classfierResult != datingLables[i]:
                errorCount += 1
        errRate = errorCount/float(numTestVecs)
        print 'final error rate is: %f' % errRate
        return errRate

    def classfiyPerson(self):
        # labels in file is 1, 2, 3
        resultList = ['not at all', 'in small doses', 'in large doses']
        percentTats = float(raw_input("percentage of time spent playing video games?"))
        ffMiles = float(raw_input('frequent flier miles earned per year?'))
        iceCream = float(raw_input('liters of ice cream consumed per year?'))
        datingDataMat, datingLabels = self.file2matrix('testDating.txt')
        normalMat, ranges, minvals = self.autoNorm(datingDataMat)
        inArr = array([ffMiles, percentTats, iceCream])
        classifierResult = self.classify0((inArr-minVals)/ranges, normalMat, datingLabels, self.k)
        print 'you will probably like this person: ', resultList[classifierResult-1]

if __name__ == '__main__':
    knn = kNN(3, 3)

    print knn.k, 'this is for data in program'
    dataSet, group = knn.createDataSet()
    knn.classify0([1,0], dataSet, group, knn.k)

    print 'this is data from file'
    dataSet, group = knn.file2matrix('testDating.txt')
    print dataSet,group

    normalDataSet, ranges, minVals = knn.autoNorm(dataSet)
    print "normal dataset", normalDataSet, ranges, minVals
    knn.classify0([5000, 10, 0.5], normalDataSet, group, knn.k)

    print 'verify the algorithm ...'
    knn.datingClassTest()

    knn.classfiyPerson()
