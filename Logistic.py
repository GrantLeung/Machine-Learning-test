import matplotlib.pyplot as plt
import numpy as np

"""
函数说明:梯度上升算法测试函数
 
求函数f(x) = -x^2 + 4x的极大值
 
Parameters:
    无
Returns:
    无
"""
def Gradient_Ascent_test():
    def f_prime(x_old):
        return -2 * x_old + 4
    x_old = -1
    x_new = 0
    alpha = 0.01
    presision = 0.00000001
    while abs(x_new - x_old) > presision:
        x_old = x_new
        x_new = x_old + alpha * f_prime(x_old)
    print(x_new)

"""
函数说明:加载数据
 
Parameters:
    无
Returns:
    dataMat - 数据列表
    labelMat - 标签列表
Author:
    Jack Cui
"""
def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('./Logistic/testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    fr.close()
    return dataMat, labelMat

"""
函数说明:绘制数据集
 
Parameters:
    无
Returns:
    无
"""
def plotDataSet():
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataMat)[0]
    xcord1 = [];ycord1 = []
    xcord2 = [];ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)                                            #添加subplot
    ax.scatter(xcord1, ycord1, s = 20, c = 'red', marker = 's',alpha=.5)#绘制正样本
    ax.scatter(xcord2, ycord2, s = 20, c = 'green',alpha=.5)            #绘制负样本
    plt.title('DataSet')                                                #绘制title
    plt.xlabel('x'); plt.ylabel('y')                                    #绘制label
    plt.show()    

if __name__ == '__main__':
    plotDataSet()