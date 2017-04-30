# -*- encoding: utf-8 -*-

from numpy import *
from numpy import linalg as la

#相似度计算，欧式距离，对数据量级敏感
def eulidSim(inA, inB):
    return 1.0/(1.0 + la.norm(inA - inB))

#皮尔逊相关系数，相对于欧氏距离，对用户的评级的量级不敏感
def pearsSim(inA, inB):
    if len(inA) < 3:
        return 1.0
    return 0.5 + 0.5*corrcoef(inA, inB, rowvar=0)[0][1]

#余弦相似度
def cosSim(inA, inB):
    num = float(inA.T * inB)
    denom = la.norm(inA) * la.norm(inB)
    return 0.5 + 0.5*(num/denom)

#估计用户user对物品item的评分,喜爱程度
def standEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0:
            continue
        #寻找两个物品都评级的用户
        overLap = nonzero(logical_and(dataMat[:,item].A>0,
                                      dataMat[:,j].A>0))[0]
        if len(overLap) == 0:
            similarity = 0
        else:
            #计算两个物品之间的相似度
            similarity = simMeas(dataMat[overLap, item], dataMat[overLap,j])
        simTotal += similarity
        #用相似度作为评分的权重，乘以对相似物品的评分，来计算待评物品分数
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal/simTotal

def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    #寻找未评级的物品
    unratedItems = nonzero(dataMat[user,:].A == 0)[1]
    if len(unratedItems) == 0:
        return 'you rated everything'
    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append((item, estimatedScore))
    #返回用户可能评分最高的N个物品
    return sorted(itemScores, key=lambda jj:jj[1], reverse=True)[:N]

#利用SVD进行评分估计
def svdEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    #奇异值分解
    U,Sigma,VT = la.svd(dataMat)
    Sig4 = mat(eye(4) * Sigma[:4])
    # 构建转换后的物品，dataMat * V = U * Sigma, 即对列（物品）进行压缩，这里为什么还要用dataMat.T 乘以U[:,:4] * Sig4.I ？？？
    xformedItems = dataMat.T * U[:,:4] * Sig4.I     #？ U[:,:4] * Sig4.I，选择4个主要物品进行相似度计算
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0 or j==item:
            continue
        similarity = simMeas(xformedItems[item,:].T, xformedItems[j,:].T)
        print 'the %d and %d similarity is: %f' % (item, j, similarity)
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal/simTotal

#利用SVD对图像进行压缩，将奇异值去掉较小的那些值，仍然可以高度精确地重建原始矩阵
def test():
    a = [[16,2,3,4,5],[2,3,4,51,6],[3,4,50,6,7],[4,5,6,7,8],[5,6,7,8,90]]
    m = mat(a)
    print m
    U,Sigma,VT = la.svd(m)
    print U
    print Sigma
    print VT
    sig4 = mat(eye(4)*Sigma[:4])
    sigRecon = mat(zeros((4,4)))
    for k in range(4):
        sigRecon[k,k] = Sigma[k]
    #print "reconSigma: \n", sigRecon
    xform = m.T * U[:,:4] * sig4.I
    print "xform: \n", xform, "\nanother xform:\n", U[:,:4] * sig4.I
    reconM = U[:,:4]*sigRecon*VT[:4,:]
    print "reconM: \n", reconM