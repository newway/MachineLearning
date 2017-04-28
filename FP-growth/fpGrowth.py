# -*- encoding: utf-8 -*-

from numpy import *

class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None        #连接不同分支下面的相同元素
        self.parent = parentNode
        self.children = {}

    def inc(self, numOccur):
        self.count += numOccur

    def disp(self, ind=1):
        print ' '*ind, self.name, " ", self.count
        for child in self.children.values():
            child.disp(ind+1)

#元素项按降序排列，更新树时从根往下匹配，出现频率最高的元素，越靠近树根，增加子节点计数或者创建新的子节点
#从树根到每个叶节点，都是一个频繁项集
def createTree(dataSet, minSup=1):
    #头指针表，存储了各个元素的总计出现次数
    headerTable = {}
    for trans in dataSet:
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]

    #移除不满足最小支持度的项
    for k in headerTable.keys():
        if headerTable[k] < minSup:
            del(headerTable[k])

    freqItemSet = set(headerTable.keys())
    if len(freqItemSet) == 0:
        return None, None
    for k in headerTable:
        headerTable[k] = [headerTable[k], None]     #[计数值， 指针]
    retTree = treeNode('Null Set', 1, None)
    for tranSet, count in dataSet.items():
        localD = {}
        for item in tranSet:
            if item in freqItemSet:
                localD[item] = headerTable[item][0]
        if len(localD) > 0:
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p:p[1], reverse=True)]
            updateTree(orderedItems, retTree, headerTable, count)
    return retTree, headerTable

def updateTree(items, inTree, headerTable, count):
    if items[0] in inTree.children:
        inTree.children[items[0]].inc(count)
    else:
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        #如果创建了新节点，需要更新相同元素的链表
        if headerTable[items[0]][1] == None:
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    if len(items) > 1:
        updateTree(items[1:], inTree.children[items[0]], headerTable, count)

def updateHeader(header, targetNode):
    while (header.nodeLink != None):
        header = header.nodeLink
    header.nodeLink = targetNode

def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat

def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        one = frozenset(trans)
        retDict[one] = retDict.get(one, 0) + 1
    return retDict

#发现以给定元素结尾到树根的路径，不包括树根
def ascendTree(leafNode, prefixPath):
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)

#发现以给定元素结尾的所有路径
def findPrefixPath(basePat, treeNode):
    #条件模式基（conditional pattern base）是以所查找元素为结尾的路径集合
    #每一条路径都是一条前缀路径，是介于所查找元素与树根节点之间的所有内容
    condPats = {}
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1:
            #查找节点的计数等于以该节点开始的路径计数，[1:0]从1开始不包括节点本身，构成条件模式基
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return condPats

#查找频繁项集
def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p:p[1])]
    #从出现频率最低的元素开始查找
    for basePat in bigL:
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        freqItemList.append(newFreqSet)
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
        #根据所有以相同元素为起点的条件模式基，构建条件FP-tree
        myCondTree, myHead = createTree(condPattBases, minSup)
        if myHead != None:
            #新的树是不包含basePat元素的Fp-tree，继续寻找频繁项集
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)