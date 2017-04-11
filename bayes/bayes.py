# -*- encoding:utf-8 -*-
from numpy import zeros, ones, log, array
import random

def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['myabe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1:侮辱性文字 0:正常
    return postingList, classVec

#所有单词集合
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print "the word: %s is not in my Vocabulary!" % word
    return returnVec

#朴素贝叶斯训练函数
#trainMatrix:词表特征向量组成的矩阵, trainCategory为文档对应的分类list
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    #词表的长度
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    #统计各个分类里面,词表中每个单词出现次数
    #p0Num = zeros(numWords); p1Num = zeros(numWords)
    #初值取1,防止条件概率为0,影响连乘结果
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    #统计各个分类里面,单词出现的总数
    #p0Denom = 0.0; p1Denom = 0.0
    p0Denom = 1.0*numWords
    p1Denom = 1.0*numWords
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    #分类里面,各个词汇出现概率, p1Denom = sum(p1Num)
    #p1Vect = p1Num / p1Denom
    #取对数防止小概率乘积的溢出
    p1Vect = log(p1Num / p1Denom)
    #p0Vect = p0Num / p0Denom
    p0Vect = log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive

#原理:p(ci|w) = p(w|ci)*p(ci) / p(w),假设各个单词独立, p(w|ci) = p(w0|ci)*p(w1|ci)...p(wn|ci)
#已知单词,求分类的条件概率
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    #条件概率求积等价于对数求和
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0-pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok)>2]

#first need email in directionay
def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        #垃圾邮件
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        # 非垃圾邮件
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)

    vocabList = createVocabList(docList)
    trainingSet = range(50)
    testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])

    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])

    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print "the error rate is: ", float(errorCount)/len(testSet)
    
if __name__ == '__main__':
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    print myVocabList
    #print setOfWords2Vec(myVocabList, listOPosts[0])
    #print setOfWords2Vec(myVocabList, listOPosts[3])
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))

    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    print p0V, sum(p0V)
    print p1V, sum(p1V)
    print pAb
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb)

    #testEntry = ['stupid', 'garbage', 'dalmation']
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb)