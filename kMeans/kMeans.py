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

def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))

#随机构造K个质心
def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k, n)))
    for j in range(n):
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:,j]) - minJ)
        #对每个特征，随机选择范围内的值为初始质点对应特征值
        centroids[:, j] = minJ + rangeJ * random.rand(k, 1)
    return centroids

#K-均值聚类算法， 弊端，随机选择k个质点，导致最后结果可能收敛于局部最小值
def kMeans(dataSet, k, distMeas = distEclud, createCent=randCent):
    m = shape(dataSet)[0]
    #第一列表示点所属质心编号， 第二列表示到质心的距离平方
    clusterAssment = mat(zeros((m, 2)))
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i,:] = minIndex, minDist**2
        print centroids
        #更新质心位置
        for cent in range(k):
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]
            centroids[cent, :] = mean(ptsInClust, axis=0)
    return centroids, clusterAssment

#二分K-均值聚类算法， 先由整个数据看成一个簇，然后在每个簇上进行K=2的聚类划分，选择误差最小的簇进行划分
def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))
    #完整数据的质心,list type
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    centList = [centroid0]
    for j in range(m):
        clusterAssment[j, 1] = distMeas(mat(centroid0), dataSet[j,:])**2
    while (len(centList) < k):
        lowestSSE = inf
        #每次选择一个簇，对其进行2分和不划分做误差对比
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A == i)[0], :]
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            #簇i进行划分的内部误差和SSE
            sseSplit = sum(splitClustAss[:,1])
            #所有不属于簇i的误差之和
            sseNoSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A != i)[0], 1])
            print 'sseSplit, and notSplit: ', sseSplit, sseNoSplit
            if (sseSplit + sseNoSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNoSplit
        #分裂的新簇分配簇编号
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0], 0] = len(centList)
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0], 0] = bestCentToSplit
        print 'the bestCentToSplit is: ', bestCentToSplit
        print 'the len of bestClustAss is: ', len(bestClustAss)
        centList[bestCentToSplit] = bestNewCents[0,:]
        centList.append(bestNewCents[1,:])
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0], :] = bestClustAss
    return mat(centList), clusterAssment




