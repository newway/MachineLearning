# -*- encoding: utf-8 -*-

from numpy import *

def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [map(float, line) for line in stringArr]
    return mat(datArr)

def pca(dataMat, topNfeat=9999999):
    #去平均值
    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals
    #计算协方差
    covMat = cov(meanRemoved, rowvar=0)
    #方阵的 特征值 和 特征向量
    eigVals, eigVects = linalg.eig(mat(covMat))
    eigValInd = argsort(eigVals)
    eigValInd = eigValInd[:-(topNfeat+1):-1]
    redEigVects = eigVects[:,eigValInd]
    #数据转换到最大的N个特征向量构建的空间中，数据降维
    lowDDataMat = meanRemoved * redEigVects
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat