# This program is an example for the diagnostic library
# Author: Adrien Dorise
# Date: September 2020

import detectionToolbox as diag
import numpy as np
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import sys
import matplotlib.pyplot as plt 
import time
import os




# %config InlineBackend.figure_format = 'retina'



# Timer class
class Timer():  
    def __enter__(self):  
        self.start()   
        # __enter__ must return an instance bound with the "as" keyword  
        return self  
      
    # There are other arguments to __exit__ but we don't care here  
    def __exit__(self, *args, **kwargs):   
        self.stop()  
      
    def start(self):  
        if hasattr(self, 'interval'):  
            del self.interval  
        self.start_time = time.time()  
  
    def stop(self):  
        if hasattr(self, 'start_time'):  
            self.interval = time.time() - self.start_time  
            del self.start_time # Force timer reinit  


def getLabel(labels, faultValue, windowSize):
    parameter = windowSize / 2 #the number of fault lable to be found in the timeWindow to modify all labels of time window
    
    tempLabels = labels.copy()
    for i in range(len(labels)):
        if i + windowSize > len(labels):
            tempLabels[i] = labels[i]
        else:
            count = 0
            for j in range(i,i+windowSize):
                if labels[j] == faultValue:
                    count += 1
            if count >= parameter:
                tempLabels[i] = faultValue
            else:
                tempLabels[i] = 0
    return tempLabels





def anomalyRemoval(faultyDataSet,normalDataSet,labelArray,faultValue,iteration,windowSize):
    # if(labelArray[iteration - windowSize] == faultValue):
    i = iteration
    while(labelArray[i] == faultValue):
        i-=1
    lastNormalIteration = i
    
    i = iteration
    while(labelArray[i] == faultValue):
        i+=1
    lastFaultIteration = i
    
    for j in range(lastNormalIteration,lastFaultIteration):
        faultyDataSet[j,:] = normalDataSet[j,:]
        labelArray[j] = 1
    
    return faultyDataSet, labelArray


def doPerPoint(model, data, classes, faultValue, cm, time, plot):
    interrupt = False
    with Timer() as timer:
        # faultyDataTemp = scaler.transform(testData[:,2].reshape(-1,1))
        # faultyDataTemp = diag.statExtraction(faultyDataTemp,windowSize,diagDataChoice)
        # faultyDataTemp = diagTestScale1
        faultyDataTemp = data.copy()
        # classTemp = testClassClassif.copy()
        classTemp = classes.copy()
        for i in range(len(faultyDataTemp[:,0])):
            if(interrupt == False):
                tempCM,tempPred = diag.confusionMatrixClassifier(faultyDataTemp[i,:],classTemp[i],model,faultValue=faultValue,classif=True)
                if(tempPred == -1 and classTemp[i] == 1):
                    interrupt = True
            else:
                tempCM = [1,0,0,0]
                if(classTemp[i] != 1):
                    interrupt = False
            cm = np.add(cm ,tempCM)
    time.append(timer.interval)
    if(plot):
        diag.plotClassification(faultyDataTemp,classTemp,model,figName='KNN',xlabel='X',ylabel='Y',save=save,folderPath=savePath)
    return cm,time


def doRupture(model, data, timeSeries, classes, faultValue, rupturePenalty, cm, time, plot) :
    with Timer() as timer: #Rupture
        # faultyDataTemp = scaler.transform(testData[:,2].reshape(-1,1))
        # faultyDataTemp = diag.statExtraction(faultyDataTemp,windowSize,diagDataChoice)
        # faultyDataTemp = diagTestScale1
        faultyDataTemp = data.copy()
        # classTemp = testClassClassif.copy()
        classTemp = classes.copy()
        timeRuptureAlgo,timeRuptureBreakPoints,timeRuptureDataBreak = diag.timeRupture(timeSeries[:,2],penaltyValue=rupturePenalty,plot=plot)
        for i in range(len(timeRuptureBreakPoints)-1):
            indices1 = [timeRuptureBreakPoints[i],timeRuptureBreakPoints[i+1]]   
            classValue1 = diag.getClass(indices1,classTemp)   #getting the real class for the set of points
            points = np.array(faultyDataTemp[indices1,0].mean(),ndmin=2)
            for featureColumn in range(1,faultyDataTemp.shape[1]):
                points = np.append(points,np.array(faultyDataTemp[indices1,featureColumn].mean(),ndmin=2),axis=1)
            tempCM,tempPred = diag.confusionMatrixClassifier(points,classValue1,model,faultValue=faultValue,classif=True)
            cm = np.add(cm ,tempCM)
    time.append(timer.interval) 
    return cm,time
    


#General parameters
# dataPath = 'H:\\DIAG_RAD\\Programs\\Diagnostic_python\\DiagnosticExample\\ExampleDataSets'
# savePath = 'H:\\DIAG_RAD\\Results\\Diagnostic\\Example\\example1' 
dataChoice = 1 #1 for simulated data: 2 for real datas
trainDataPath = "H:\\DIAG_RAD\\DataSets\\\endThesisValidationData\\simulations\\trainSet\\microLatch"
testDataPath = "H:\\DIAG_RAD\\DataSets\\\endThesisValidationData\\simulations\\testSet\\microLatch"
savePathFolder = 'H:\\DIAG_RAD\\Results\\IFAC_Safeprocess_2021\\multiple_testSets\\3classes\\mean_variance_trainSet'
resultPath = 'H:\\DIAG_RAD\\Results\\IFAC_Safeprocess_2021\\Accuracy\\classificationAllStats\\classi2'
saveResult = 0

doTestParam = False
doTestClassif = True
diagDataChoice = 6 # 1 (mean & variance); 2 (mean & frequency); 3 (variance & frequency); 4 (mean & min & max & variance & skewness & kurtosis); 5 (mean & min & max & variance & skewness & kurtosis & freq); 6 all stats

addNewLatchClass = 0
bigLatch = False #Adjust the labels in the time window
windowSize = 20
rupturePenalty = 0.8
faultValue = 1



trainParamRange =  range(1,5+1)
testParamRange = range(1,10+1)
trainRange = range(1,10+1)
testRange = range(1,20+1)


plotFeatures = 0
plotTimeRupt = 0
plotClassification = 0
plotDiagRupture = 0
plotDiagPerPoints = 0
plotAccuracyPerMethods = 0
plotAccuracyAllSets = 0


save = 0



if save == 1 or saveResult == 1:
    inp = input('The parameter save equal 1. Do you really wish to save the datas (might overwritte older files)?\nY or N ?\n')
    if (inp == 'N' or inp == 'n' or inp == 'no' or inp == 'No'):
        sys.exit()



scaler = MinMaxScaler(feature_range=(0,1))
data=pd.DataFrame()
classificationList = ['OCSVM', 'elliptic classification', 'LOF', 'isolation forest']
classificationName = ['OCSVM', 'Elliptic classification', 'LOF', 'Isolation forest']
colors = ['green','red','blue','blue','blue','blue','blue','blue','blue','blue',]
className = ['normal','latch','','','','latch','front de latch up']

#Classification

#Finding Best parameters for each algorithm


kParam = [1,2,3,4,5,6,7,8,9,20,50,100,7000]
forestParam=[10,25,50,75,100,125,150,175,200,225,250,275,300]
accuracyParamKnnPerPoint = np.zeros((len(trainParamRange),len(kParam)))
accuracyParamKnnRupt = np.zeros((len(trainParamRange),len(kParam)))
accuracyParamForestPerPoint = np.zeros((len(trainParamRange),len(forestParam)))
accuracyParamForestRupt = np.zeros((len(trainParamRange),len(forestParam)))





if doTestParam:
    cmPerPointKnn,cmRuptKnn = [0,0,0,0], [0,0,0,0] #TP,FP,FN,TN
    cmPerPointForest,cmRuptForest = [0,0,0,0], [0,0,0,0] #TP,FP,FN,TN
        
    timePerPointKnn,timeRuptKnn = [], [] 
    timePerPointForest,timeRuptForest = [], [] 
    
    for trainSetNumber in trainParamRange:
        savePath = savePathFolder + str(trainSetNumber)
        data=data.append(diag.ifacDataFrame('train'+str(trainSetNumber)))
        trainData, diagTrain1,diagTrainScale1,trainClass, featureChoice, xlabel,ylabel= diag.preprocessing(dataPath = trainDataPath, dataIndice = trainSetNumber,dataChoice = dataChoice,windowSize=windowSize, dataName = 'trainSet',diagDataChoice = diagDataChoice,plotFeatures = plotFeatures,save=save)
        trainClass[np.where(trainClass == 5)[0]] = 1 
        trainScale = scaler.fit_transform(trainData[:,2].reshape(-1,1))
        # diagTrainScale1 = diag.statExtraction(trainScale,windowSize,diagDataChoice)
        if(plotFeatures):
            diag.plotLabelPoints(diagTrainScale1, trainClass, className,figName='trainSet',colors=colors,xlabel=xlabel,ylabel=ylabel,save=save,folderPath=savePath)
        
        if bigLatch:
            trainClass = getLabel(trainClass,faultValue=faultValue,windowSize=windowSize)
            
            
        accuracyTemp1=np.zeros((len(testParamRange),len(kParam)))
        accuracyTemp2=np.zeros((len(testParamRange),len(kParam)))
        accuracyTemp3=np.zeros((len(testParamRange),len(forestParam)))
        accuracyTemp4=np.zeros((len(testParamRange),len(forestParam)))
        
        
        
        for testSetNumber in testParamRange:
            
            #Import Data
            testData, diagTest1 ,diagTestScale1,testClass,featureChoice, xlabel,ylabel = diag.preprocessing(dataColumnChoice = 2, dataPath = testDataPath, windowSize=windowSize, dataIndice = testSetNumber,dataChoice = dataChoice, dataName = ' ',diagDataChoice = diagDataChoice,plotFeatures = plotFeatures,save=save)
            featureName = featureChoice
            faultySetScale = scaler.transform(testData[:,2].reshape(-1,1))
            normalSet = scaler.transform(testData[:,1].reshape(-1,1))
            diagNormalScale = diag.statExtraction(normalSet,windowSize,diagDataChoice)
            testClassClassif = testClass.copy()
            testClassClassif[np.where(testClassClassif != 5)[0]] = 0
            testClassClassif[np.where(testClassClassif == 5)[0]] = 1

            
    
            # We do a classification for all algorithms possible in order to compare them
            classifierChoice = 'Knn'
            i=0
            for k in kParam:
                modelKnn,trainRoc1,TrainCm1,trainCmAcc1 = diag.classifier(diagTrainScale1,trainClass,'Knn',knn_n_neighbors=k,figName='Train_Classif_'+featureChoice,plot=plotClassification,classesName=className,xlabel=xlabel,ylabel=ylabel,save=save,folderPath=savePath)
                cmPerPointKnn,timePerPointKnn = doPerPoint(modelKnn, diagTestScale1, testClassClassif, faultValue,cmPerPointKnn,timePerPointKnn, plotDiagPerPoints)
                cmRuptKnn,timeRuptKnn = doRupture(modelKnn, diagTestScale1, testData, testClassClassif, faultValue, rupturePenalty, cmRuptKnn, timeRuptKnn, plotDiagPerPoints)
                
                accuracyPerPoint = (cmPerPointKnn[0] + cmPerPointKnn[3]) / (cmPerPointKnn[0] + cmPerPointKnn[1] + cmPerPointKnn[2] + cmPerPointKnn[3])
                accuracyRupture = (cmRuptKnn[0] + cmRuptKnn[3]) / (cmRuptKnn[0] + cmRuptKnn[1] + cmRuptKnn[2] + cmRuptKnn[3])
                accuracyTemp1[testSetNumber-1,i] = accuracyPerPoint
                accuracyTemp2[testSetNumber-1,i] = accuracyRupture
                i=i+1
                
                    # We do a classification for all algorithms possible in order to compare them
            classifierChoice = 'random_forest_classifier'
            i=0
            for ensemble in forestParam:
                modelForest,trainRoc1,TrainCm1,trainCmAcc1 = diag.classifier(diagTrainScale1,trainClass,'random_forest_classifier',ensemble_estimators=ensemble,figName='Train_Classif_'+featureChoice,plot=plotClassification,classesName=className,xlabel=xlabel,ylabel=ylabel,save=save,folderPath=savePath)
                cmPerPointForest,timePerPointForest = doPerPoint(modelForest, diagTestScale1, testClassClassif, faultValue,cmPerPointForest,timePerPointForest, plotDiagPerPoints)
                cmRuptForest,timeRuptForest = doRupture(modelForest, diagTestScale1, testData, testClassClassif, faultValue, rupturePenalty, cmRuptForest, timeRuptForest, plotDiagPerPoints)
            
                accuracyPerPoint = (cmPerPointForest[0] + cmPerPointForest[3]) / (cmPerPointForest[0] + cmPerPointForest[1] + cmPerPointForest[2] + cmPerPointForest[3])
                accuracyRupture = (cmRuptForest[0] + cmRuptForest[3]) / (cmRuptForest[0] + cmRuptForest[1] + cmRuptForest[2] + cmRuptForest[3])
                accuracyTemp3[testSetNumber-1,i] = accuracyPerPoint
                accuracyTemp4[testSetNumber-1,i] = accuracyRupture
                i=i+1
                
            print('\nPARAMETERS for trainSet ' + str(trainSetNumber) + ' in test set ' + str(testSetNumber) +'\n')
        #We complete accuracy matrix
        for i in range(len(accuracyTemp1[0,:])):
            accuracyParamKnnPerPoint[trainSetNumber-1,i] = np.mean(accuracyTemp1[:,i])
            accuracyParamKnnRupt[trainSetNumber-1,i] = np.mean(accuracyTemp2[:,i])
            accuracyParamForestPerPoint[trainSetNumber-1,i] = np.mean(accuracyTemp3[:,i])
            accuracyParamForestRupt[trainSetNumber-1,i] = np.mean(accuracyTemp4[:,i])
        
    
    print("\nTrain set: " + trainDataPath)
    print("Test set: " + testDataPath)
    
    print("\nMean accuracy parameters KNN")
    print(kParam)
    for i in range(len(trainParamRange)):
        print("PerPoint: " + str(accuracyParamKnnPerPoint[i]))
        print("Rupture: " + str(accuracyParamKnnRupt[i]))
    
    print("\nMean accuracy parameters Rand forest")
    print(forestParam)
    for i in range(len(trainParamRange)):
        print("PerPoint: " + str(accuracyParamForestPerPoint[i]))
        print("Rupture: " + str(accuracyParamForestRupt[i]))
    
      
    
    
    
    
    
    
    
    

if doTestClassif:
    #They are numpy array with lines = testSets and columns = trainSets
    accuracyPerPointKnn,accuracyRuptureKnn,accuracyPerPointBayes,accuracyRuptureBayes,accuracyPerPointTree,accuracyRuptureTree,accuracyPerPointForest,accuracyRuptureForest,accuracyPerPointSVM,accuracyRuptureSVM=np.zeros((len(testRange),len(trainRange))),np.zeros((len(testRange),len(trainRange))),np.zeros((len(testRange),len(trainRange))),np.zeros((len(testRange),len(trainRange))),np.zeros((len(testRange),len(trainRange))),np.zeros((len(testRange),len(trainRange))),np.zeros((len(testRange),len(trainRange))),np.zeros((len(testRange),len(trainRange))),np.zeros((len(testRange),len(trainRange))),np.zeros((len(testRange),len(trainRange)))
    
    cmPerPointSVM,cmRuptSVM = [0,0,0,0], [0,0,0,0] #TP,FP,FN,TN
    cmPerPointKnn,cmRuptKnn = [0,0,0,0], [0,0,0,0] #TP,FP,FN,TN
    cmPerPointBayes,cmRuptBayes = [0,0,0,0], [0,0,0,0] #TP,FP,FN,TN
    cmPerPointTree,cmRuptTree = [0,0,0,0], [0,0,0,0] #TP,FP,FN,TN
    cmPerPointForest,cmRuptForest = [0,0,0,0], [0,0,0,0] #TP,FP,FN,TN
    
    timePerPointSVM,timeRuptSVM = [], [] 
    timePerPointKnn,timeRuptKnn = [], [] 
    timePerPointBayes,timeRuptBayes = [], []
    timePerPointTree,timeRuptTree = [], []
    timePerPointForest,timeRuptForest = [], [] 
    
    
    
    #Accuracy comparison for all methods
    for trainSetNumber in trainRange:
        savePath = savePathFolder + str(trainSetNumber)
        data=data.append(diag.ifacDataFrame('train'+str(trainSetNumber)))
        trainData, diagTrain1,diagTrainScale1,trainClass, featureChoice, xlabel,ylabel= diag.preprocessing(dataPath = trainDataPath, dataIndice = trainSetNumber,dataChoice = dataChoice,windowSize=windowSize, dataName = 'trainSet',diagDataChoice = diagDataChoice,plotFeatures = plotFeatures,save=save)
        trainClass[np.where(trainClass == 5)[0]] = 1 
        trainScale = scaler.fit_transform(trainData[:,2].reshape(-1,1))
        # diagTrainScale1 = diag.statExtraction(trainScale,windowSize,diagDataChoice)
        if(plotFeatures):
            diag.plotLabelPoints(diagTrainScale1, trainClass, className,figName='trainSet',colors=colors,xlabel=xlabel,ylabel=ylabel,save=save,folderPath=savePath)
        
        if bigLatch:
            trainClass = getLabel(trainClass,faultValue=faultValue,windowSize=windowSize)
        
        k=9
        modelKnn,trainRoc1,TrainCm1,trainCmAcc1 = diag.classifier(diagTrainScale1,trainClass,'Knn',knn_n_neighbors=k,figName='Train_Classif_'+featureChoice,plot=plotClassification,classesName=className,xlabel=xlabel,ylabel=ylabel,save=save,folderPath=savePath)
        modelBayes,trainRoc1,TrainCm1,trainCmAcc1 = diag.classifier(diagTrainScale1,trainClass,'naive_bayes',figName='Train_Classif_'+featureChoice,plot=plotClassification,classesName=className,xlabel=xlabel,ylabel=ylabel,save=save,folderPath=savePath)
        modelTree,trainRoc1,TrainCm1,trainCmAcc1 = diag.classifier(diagTrainScale1,trainClass,'decision_tree_classifier',figName='Train_Classif_'+featureChoice,plot=plotClassification,classesName=className,xlabel=xlabel,ylabel=ylabel,save=save,folderPath=savePath)
        ensemble = 50
        modelForest,trainRoc1,TrainCm1,trainCmAcc1 = diag.classifier(diagTrainScale1,trainClass,'random_forest_classifier',ensemble_estimators=ensemble,figName='Train_Classif_'+featureChoice,plot=plotClassification,classesName=className,xlabel=xlabel,ylabel=ylabel,save=save,folderPath=savePath)
        modelSVM,trainRoc1,TrainCm1,trainCmAcc1 = diag.classifier(diagTrainScale1,trainClass,'svm',figName='Train_Classif_'+featureChoice,plot=plotClassification,classesName=className,xlabel=xlabel,ylabel=ylabel,save=save,folderPath=savePath)
        
        
        for testSetNumber in testRange:
            
            #Import Data
            testData, diagTest1 ,diagTestScale1,testClass,featureChoice, xlabel,ylabel = diag.preprocessing(dataPath = testDataPath, windowSize=windowSize, dataIndice = testSetNumber,dataChoice = dataChoice, dataName = ' ',diagDataChoice = diagDataChoice,plotFeatures = plotFeatures,save=save)
            featureName = featureChoice
            faultySetScale = scaler.transform(testData[:,2].reshape(-1,1))
            normalSet = scaler.transform(testData[:,1].reshape(-1,1))
            # diagNormalScale = diag.statExtraction(normalSet,windowSize,diagDataChoice)
            testClassClassif = testClass.copy()
            testClassClassif[np.where(testClassClassif != 5)[0]] = 0
            testClassClassif[np.where(testClassClassif == 5)[0]] = 1
            
            #!!!!!!This function is to be used when the fault value is too important and mess up witht the sliding window!!!!!!!!!!!!!!!!!!!!!
            # if bigLatch:
            #     testClassClassif = getLabel(testClassClassif,faultValue=1,windowSize=windowSize)
           
            
            
            # We do a classification for all algorithms possible in order to compare them
            classifierChoice = 'Knn'
            cmPerPointKnn,timePerPointKnn = doPerPoint(modelKnn, diagTestScale1, testClassClassif, faultValue,cmPerPointKnn,timePerPointKnn, plotDiagPerPoints)
            cmRuptKnn,timeRuptKnn = doRupture(modelKnn, diagTestScale1, testData, testClassClassif, faultValue, rupturePenalty, cmRuptKnn, timeRuptKnn, plotDiagPerPoints)
            
           
    
            classifierChoice = 'naive_bayes'
            cmPerPointBayes,timePerPointBayes = doPerPoint(modelBayes, diagTestScale1, testClassClassif, faultValue,cmPerPointKnn,timePerPointKnn, plotDiagPerPoints)
            cmRuptBayes,timeRuptBayes = doRupture(modelBayes, diagTestScale1, testData, testClassClassif, faultValue, rupturePenalty, cmRuptKnn, timeRuptKnn, plotDiagPerPoints)
            
            
    
            classifierChoice = 'decision_tree_classifier'
            cmPerPointTree,timePerPointTree = doPerPoint(modelTree, diagTestScale1, testClassClassif, faultValue,cmPerPointTree,timePerPointTree, plotDiagPerPoints)
            cmRuptTree,timeRuptTree = doRupture(modelTree, diagTestScale1, testData, testClassClassif, faultValue, rupturePenalty, cmRuptTree, timeRuptTree, plotDiagPerPoints)
            
    
    
            classifierChoice = 'random_forest_classifier'
            cmPerPointForest,timePerPointForest = doPerPoint(modelForest, diagTestScale1, testClassClassif, faultValue,cmPerPointForest,timePerPointForest, plotDiagPerPoints)
            cmRuptForest,timeRuptForest = doRupture(modelForest, diagTestScale1, testData, testClassClassif, faultValue, rupturePenalty, cmRuptForest, timeRuptForest, plotDiagPerPoints)
            
    
            classifierChoice = 'svm'
            cmPerPointSVM,timePerPointSVM = doPerPoint(modelSVM, diagTestScale1, testClassClassif, faultValue,cmPerPointSVM,timePerPointSVM, plotDiagPerPoints)
            cmRuptSVM,timeRuptSVM = doRupture(modelSVM, diagTestScale1, testData, testClassClassif, faultValue, rupturePenalty, cmRuptSVM, timeRuptSVM, plotDiagPerPoints)
            
            
            print('\nACCURACY for trainSet ' + str(trainSetNumber) + ' in test set ' + str(testSetNumber) + '\n')
    
    
    
    
    
    #Results
        
    totalPerPoint = cmPerPointSVM[0] + cmPerPointSVM[1] + cmPerPointSVM[2] + cmPerPointSVM[3]
    
    
    cmPercentPerPointKnn= [100*cmPerPointKnn[0]/totalPerPoint, 100*cmPerPointKnn[1]/totalPerPoint, 100*cmPerPointKnn[2]/totalPerPoint, 100*cmPerPointKnn[3]/totalPerPoint]
    cmPercentPerPointBayes = [100*cmPerPointBayes[0]/totalPerPoint, 100*cmPerPointBayes[1]/totalPerPoint, 100*cmPerPointBayes[2]/totalPerPoint, 100*cmPerPointBayes[3]/totalPerPoint]
    cmPercentPerPointTree = [100*cmPerPointTree[0]/totalPerPoint, 100*cmPerPointTree[1]/totalPerPoint, 100*cmPerPointTree[2]/totalPerPoint, 100*cmPerPointTree[3]/totalPerPoint]
    cmPercentPerPointForest = [100*cmPerPointForest[0]/totalPerPoint, 100*cmPerPointForest[1]/totalPerPoint, 100*cmPerPointForest[2]/totalPerPoint, 100*cmPerPointForest[3]/totalPerPoint]
    cmPercentPerPointSVM = [100*cmPerPointSVM[0]/totalPerPoint, 100*cmPerPointSVM[1]/totalPerPoint, 100*cmPerPointSVM[2]/totalPerPoint, 100*cmPerPointSVM[3]/totalPerPoint]
    
    totalPointRupture = cmRuptKnn[0] + cmRuptKnn[1] + cmRuptKnn[2] + cmRuptKnn[3]
    cmPercentRuptKnn= [100*cmRuptKnn[0]/totalPointRupture, 100*cmRuptKnn[1]/totalPointRupture, 100*cmRuptKnn[2]/totalPointRupture, 100*cmRuptKnn[3]/totalPointRupture]
    totalPointRupture = cmRuptBayes[0] + cmRuptBayes[1] + cmRuptBayes[2] + cmRuptBayes[3]
    cmPercentRuptBayes = [100*cmRuptBayes[0]/totalPointRupture, 100*cmRuptBayes[1]/totalPointRupture, 100*cmRuptBayes[2]/totalPointRupture, 100*cmRuptBayes[3]/totalPointRupture]
    totalPointRupture = cmRuptTree[0] + cmRuptTree[1] + cmRuptTree[2] + cmRuptTree[3]
    cmPercentRuptTree = [100*cmRuptTree[0]/totalPointRupture, 100*cmRuptTree[1]/totalPointRupture, 100*cmRuptTree[2]/totalPointRupture, 100*cmRuptTree[3]/totalPointRupture]
    totalPointRupture = cmRuptForest[0] + cmRuptForest[1] + cmRuptForest[2] + cmRuptForest[3]
    cmPercentRuptForest = [100*cmRuptForest[0]/totalPointRupture, 100*cmRuptForest[1]/totalPointRupture, 100*cmRuptForest[2]/totalPointRupture, 100*cmRuptForest[3]/totalPointRupture]
    totalPointRupture = cmRuptSVM[0] + cmRuptSVM[1] + cmRuptSVM[2] + cmRuptSVM[3]
    cmPercentRuptSVM = [100*cmRuptSVM[0]/totalPointRupture, 100*cmRuptSVM[1]/totalPointRupture, 100*cmRuptSVM[2]/totalPointRupture, 100*cmRuptSVM[3]/totalPointRupture]
    
    
    
    
    meanTimePerPointKnn = sum(timePerPointKnn)/len(timePerPointKnn)
    meanTimeRuptKnn = sum(timeRuptKnn)/len(timeRuptKnn)
    meanTimePerPointBayes = sum(timePerPointBayes)/len(timePerPointBayes)
    meanTimeRuptBayes = sum(timeRuptBayes)/len(timeRuptBayes)
    meanTimePerPointTree = sum(timePerPointTree)/len(timePerPointTree)
    meanTimeRuptTree = sum(timeRuptTree)/len(timeRuptTree)
    meanTimePerPointForest = sum(timePerPointForest)/len(timePerPointForest)
    meanTimeRuptForest = sum(timeRuptForest)/len(timeRuptForest)
    meanTimePerPointSVM = sum(timePerPointSVM)/len(timePerPointSVM)
    meanTimeRuptSVM = sum(timeRuptSVM)/len(timeRuptSVM)
    
    print("Train set: " + trainDataPath)
    print("Test set: " + testDataPath)
    print('windowSize: ' + str(windowSize) + ' / RuptPenalty: ' + str(rupturePenalty)) 
    print("\nTP FP FN TN")
    
    
    print("KNN:")
    print('k: ' + str(k))
    print("PerPoint")
    print(cmPerPointKnn)
    print(cmPercentPerPointKnn)
    print("mean time: " + str(meanTimePerPointKnn))
    print("Rupture:")
    print(cmRuptKnn)
    print(cmPercentRuptKnn)
    print("mean time: " + str(meanTimeRuptKnn))
    
    
    print("\nBayes:")
    print("PerPoint")
    print(cmPerPointBayes)
    print(cmPercentPerPointBayes)
    print("mean time: " + str(meanTimePerPointBayes))
    print("Rupture:")
    print(cmRuptBayes)
    print(cmPercentRuptBayes)
    print("mean time: " + str(meanTimeRuptBayes))
    
    
    print("\nTree:")
    print("PerPoint")
    print(cmPerPointTree)
    print(cmPercentPerPointTree)
    print("mean time: " + str(meanTimePerPointTree))
    print("Rupture:")
    print(cmRuptTree)
    print(cmPercentRuptTree)
    print("mean time: " + str(meanTimeRuptTree))
    
    print("\nForest:")
    print('ensemble: ' + str(ensemble))
    print("PerPoint")
    print(cmPerPointForest)
    print(cmPercentPerPointForest)
    print("mean time: " + str(meanTimePerPointForest))
    print("Rupture:")
    print(cmRuptForest)
    print(cmPercentRuptForest)
    print("mean time: " + str(meanTimePerPointForest))
    
    
    print("\nSVM:")
    print("PerPoint")
    print(cmPerPointSVM)
    print(cmPercentPerPointSVM)
    print("mean time: " + str(meanTimePerPointSVM))
    print("Rupture:")
    print(cmRuptSVM)
    print(cmPercentRuptSVM)
    print("mean time: " + str(meanTimeRuptSVM))
    
    
    
    
