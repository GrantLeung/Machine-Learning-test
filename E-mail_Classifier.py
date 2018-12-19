# -*- coding: UTF-8 -*-
import numpy as np
import random
import re

"""
函数说明:根据vocabList词汇表，将inputSet向量化，向量的每个元素为1或0
 
Parameters:
    vocabList - createVocabList返回的列表
    inputSet - 切分的词条列表
Returns:
    returnVec - 文档向量,词集模型
"""
def setOfWords2Vec(vocabList, inputSet):
    # 创建一个其中所含元素都为0的向量
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else : print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

"""
函数说明:根据vocabList词汇表，构建词袋模型
 
Parameters:
    vocabList - createVocabList返回的列表
    inputSet - 切分的词条列表
Returns:
    returnVec - 文档向.量,词袋模型
"""
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

"""
函数说明:接收一个大字符串并将其解析为字符串列表
 
Parameters:
    无
Returns:
    无
"""
def textParse(bigString):
    listOfTokens = re.split(r'\W', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

"""
函数说明:将切分的实验样本词条整理成不重复的词条列表，也就是词汇表
 
Parameters:
    dataSet - 整理的样本数据集
Returns:
    vocabSet - 返回不重复的词条列表，也就是词汇表
"""
def createVocaBList(dataSet):
    # 创建一个空的不重复集合
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


if __name__ == '__main__':
    docList = []
    classList = []
    for i in range(1,26):
        wordList = textParse(open("email/spam/%d.txt" %i, 'r', encoding='ISO-8859-1').read())
        docList.append(wordList)
        classList.append(1)
        wordList = textParse(open("email/ham/%d.txt" %i, 'r', encoding='ISO-8859-1').read())
        docList.append(wordList)
        classList.append(0)
    vocabList = createVocaBList(docList)
    print(vocabList)
