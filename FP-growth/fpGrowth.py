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

