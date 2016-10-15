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
        diffMat = tile(inX, (dataSetSize,1)) - dataSet
        sqDiffMat = diffMat ** 2
        sqDistances = sqDiffMat.sum(axis=1)
        distances = sqDistances ** 0.5
        sortedDistIndicies = distances.argsort()
        classCount = {}
        for i in range(k):
            ithVoteclass = labels[sortedDistIndicies[i]]
            classCount[ithVoteclass] = classCount.get(ithVoteclass, 0) + 1

        #sortedClassCount = sorted(classCount.iteritems(), key=lambda x:x[1], reverse=True)
        sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
        print sortedClassCount[0][0]
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



if __name__ == '__main__':
    knn = kNN(3, 3)
    print knn.k, 'this is for data in program'
    dataSet, group = knn.createDataSet()
    knn.classify0([1,0], dataSet, group, knn.k)

    print 'this is data from file'
    dataSet, group = knn.file2matrix('testDating.txt')
    print dataSet,group
    knn.classify0([5000, 10, 0.5], dataSet, group, knn.k)

