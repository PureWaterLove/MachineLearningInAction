#coding=utf-8

from numpy import * # 科学计算包numpy
from os import listdir
import operator     #运算符模块
import matplotlib
import matplotlib.pyplot as plt


#from imp import reload

#测试数据
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

#分类函数
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]

    #距离计算 开始
    diffMat = tile(inX,(dataSetSize,1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis = 1)
    distances = sqDistances ** 0.5
    #距离计算 结束

    sortedDistIndicies = distances.argsort()
    classCount = {}

    #选择距离最小的k个点 开始
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    #选择距离最小的k个点 结束


    sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1), reverse = True) #排序
    return sortedClassCount[0][0]

#文本记录解析程序
def file2matrix(filename):

    #打开文件并获取文件有多少行 开始
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    #打开文件并获取文件有多少行 结束

    #创建返回的NumPy矩阵
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    index = 0

    #解析文件数据到列表 开始
    for line in arrayOLines:
        line = line.strip() #截取掉回车字符
        listFromLine = line.split('\t') #用 \t 将上一步得到的整行数据分割成一个元素列表
        returnMat[index,:] = listFromLine[0:3] #选取前3个元素，并存储到特征矩阵中
        classLabelVector.append(int(listFromLine[-1])) #索引值-1表示列表中的最后一列元素
        index += 1
    #解析文件数据到列表 结束
    return returnMat,classLabelVector

#归一化特征值
def autoNorm(dataSet):
    minVals = dataSet.min(0) #从列中选取最小值
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals,(m,1)) #tile()函数将变量内容复制成输人矩阵同样大小的矩阵
    normDataSet = normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals

#测试代码
def datingClassTest():
    hoRatio = 0.02
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        #print "the classifier came back with: %d, the real answer is : %d" % (classifierResult,datingLabels[i])
        print('the classifier came back with: {}, the real answer is : {}'.format(classifierResult,datingLabels[i]))
        if(classifierResult != datingLabels[i]):
            errorCount += 1.0
    #print "the total error rate is : %f" %(errorCount/float(numTestVecs))
    print('the total error rate is : {}'.format(errorCount/float(numTestVecs)))

#约会网站预测函数
def classifyPerson():
    resultList = ['not at all','in small doses','in large doses']
    percentTats = float(input('percentage of time spent playing video games?'))
    ffMiles = float(input('frequent flier miles earned per years?'))
    iceCream = float(input('liters of ice cream consumed per year'))
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles,percentTats,iceCream])
    classifierResult = classify0((inArr - minVals)/ranges,normMat,datingLabels,3)
    print('You will probably like this person  : {}'.format(resultList[classifierResult - 1]))

#将二进制图像转化为向量
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32 * i + j] = int(lineStr[j])
    return returnVect

#手写识别系统测试代码
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('digits/trainingDigits') #获取目录内容
    m = len(trainingFileList) #目录中文件个数
    trainingMat = zeros((m,1024)) #每行数据存储一个图像
    for i in range(m):
        #从文件名解析分类数据 开始
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumberStr = int(fileStr.split('_')[0])
        #从文件名解析分类数据 结束
        hwLabels.append(classNumberStr)
        trainingMat[i,:] = img2vector('digits/trainingDigits/%s' %fileNameStr)
    testFileList = listdir('digits/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split(',')[0]
        classNumberStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('digits/testDigits/%s' %fileNameStr)
        classifierResult = classify0(vectorUnderTest,trainingMat,hwLabels,3)
        print('the classifier came back with : {}, the real answer is : {}'.format(classifierResult,classNumberStr))
        if(classifierResult != classNumberStr):
            errorCount += 1.0
    print('\nthe total number of errors is : {}'.format(errorCount))
    print('\nthe total error rate is : {}'.format(errorCount/float(mTest)))

"""以下代码均为调试函数所用代码，需要时取出即可
#group,labels = createDataSet()

#print(classify0([0,0],group,labels,3))

#reload(kNN)

#加载数据集
#datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
#print(datingDataMat)
#print(datingLabels[0:20])

#数据集图像化
#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.scatter(datingDataMat[:,1],datingDataMat[:,2]) #无标记散点图
#ax.scatter(datingDataMat[:,1],datingDataMat[:,2],15.0*array(datingLabels),15.0*array(datingLabels)) #有标记散点图
#plt.show()

normMat10,ranges,minVals = autoNorm(datingDataMat)
print("normMat = ")
print(normMat)

print("ranges = ")
print(ranges)

print("minVals = ")
print(minVals)

#datingClassTest()

#classifyPerson()

testVector = img2vector('digits/testDigits/0_1.txt')
print(testVector[0,0:31])
print(testVector[0,32:63])

"""

handwritingClassTest()
