'''
This section will use Logistic regression to predict the survival of horses with hernias. 
Data 1 here contains 368 samples and 28 features. I am not an expert in horse breeding. 
I have learned from some literature that hernia is a term used to describe horse gastrointestinal pain. 
However, this disease does not necessarily originate from a horse's gastrointestinal problems. 
Other problems may also cause equine hernia. The data set contains some indicators for the hospital to detect horse hernia. 
Some indicators are subjective, and some are difficult to measure, such as the level of pain in horses.
'''
import logRegres
from numpy import *

def classifyVector(inX, weights):
    prob = logRegres.sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0

def colicTest():
    frTrain = open('horseColicTraining.txt'); frTest = open('horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = logRegres.stocGradAscent1(array(trainingSet), trainingLabels, 1000)
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights))!= int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print("the error rate of this test is: %f" % errorRate)
    return errorRate

def multiTest():
    numTests = 10; errorSum=0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests)))
        

multiTest()