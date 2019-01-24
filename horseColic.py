import random
import numpy as np

"""
函数说明:sigmoid函数
 
Parameters:
    inX - 数据
Returns:
    sigmoid函数
"""
def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))

"""
函数说明:改进的随机梯度上升算法
 
Parameters:
    dataMatrix - 数据数组
    classLabels - 数据标签
    numIter - 迭代次数
Returns:
    weights - 求得的回归系数数组(最优参数)
"""
def stocGradAscent1(dataMatrix, classLabels, numIter = 150):
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha *error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

"""
函数说明:梯度上升算法
 
Parameters:
    dataMatIn - 数据集
    classLabels - 数据标签
Returns:
    weights.getA() - 求得的权重数组(最优参数)
"""
def gradAscent(dataMatIn, classLabels):
    # 转换成numpy的mat
    dataMatrix = np.mat(dataMatIn)
    # 转换成numpy的mat,并进行转置
    labelMat = np.mat(classLabels).transpose() 
    # 返回dataMatrix的大小。m为行数,n为列数
    m, n = np.shape(dataMatrix)
    alpha = 0.01
    maxCycles = 500 
    weights = np.ones((n,1))
    #将矩阵转换为数组，并返回
    for k in range(maxCycles):
        #梯度上升矢量化公式
        h = sigmoid(dataMatrix * weights) 
        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights.getA()

"""
函数说明:使用Python写的Logistic分类器做预测
 
Parameters:
    无
Returns:
    无
"""
def colicTest():
    frTrain = open('./Logistic/horseColicTraining.txt')
    frTest = open('./Logistic/horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[-1]))
    # 使用改进的随即上升梯度训练
    # trainWeights = stocGradAscent1(np.array(trainingSet), trainingLabels, 500)
    trainWeights = gradAscent(np.array(trainingSet), trainingLabels)
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
        # if int(classifyVector(np.array(lineArr), trainWeights)) != int(currLine[-1]):
        if int(classifyVector(np.array(lineArr), trainWeights[:,0]))!= int(currLine[-1]):
            errorCount += 1
    # 错误率计算
    errorRate = (float(errorCount)/numTestVec) * 100
    print("测试集错误率为: %.2f%%" % errorRate)

"""
函数说明:分类函数
 
Parameters:
    inX - 特征向量
    weights - 回归系数
Returns:
    分类结果 
"""
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0

if __name__ == '__main__':
    colicTest()
