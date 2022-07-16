# This program is an example for the diagnostic library
# Author: Adrien Dorise
# Date: September 2020

import detectionToolbox as diag
import numpy as np
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
    with Timer() as timer:
        # faultyDataTemp = scaler.transform(testData[:,2].reshape(-1,1))
        # faultyDataTemp = diag.statExtraction(faultyDataTemp,windowSize,diagDataChoice)
        # faultyDataTemp = diagTestScale1
        faultyDataTemp = data.copy()
        # classTemp = testClassClassif.copy()
        classTemp = classes.copy()
        for i in range(len(faultyDataTemp[:,0])):
            tempCM,tempPred = diag.confusionMatrixClassifier(faultyDataTemp[i,:],classTemp[i],model,faultValue=faultValue,classif=True)
            # if(tempPred == -1 and classTemp[i] == 1):
            #     faultyDataTemp, classTemp = anomalyRemoval(faultyDataTemp,diagNormalScale,classTemp,1,i,windowSize)
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

diagDataChoice = 1 # 1 (mean & variance); 2 (mean & frequency); 3 (variance & frequency); 4 (mean & min & max & variance & skewness & kurtosis); 5 (mean & min & max & variance & skewness & kurtosis & freq); 6 all stats

addNewLatchClass = 0
bigLatch = False #Adjust the labels in the time window
windowSize = 10
rupturePenalty = 0.8
faultValue = 1



trainParamRange =  range(1,1+1)
testParamRange = range(1,1+1)
trainRange = range(1,1+1)
testRange = range(1,1+1)


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

#accuracyPerPoint1 is the accuracy per points for each method listed in classificationList
#accuracyPerPoint2 is the confusion matrix per points for each method listed in classificationList
#accuracyRupture gives the accuracy per point for each method
#accuracyXtotal gives total accuracy for all methods listed and for multiple training sets

kParam = [1,2,3,4,5,6,7,8,9,20,50,100,7000]
forestParam=[10,25,50,75,100,125,150,175,200,225,250,275,300]
accuracyParamKnnPerPoint = np.zeros((len(trainParamRange),len(kParam)))
accuracyParamKnnRupt = np.zeros((len(trainParamRange),len(kParam)))
accuracyParamForestPerPoint = np.zeros((len(trainParamRange),len(forestParam)))
accuracyParamForestRupt = np.zeros((len(trainParamRange),len(forestParam)))
'''

for trainSetNumber in trainParamRange:
    savePath = savePathFolder + str(trainSetNumber)
    trainDataPath = dataPath + '\\TrainLatch2'
    data=data.append(diag.ifacDataFrame('train'+str(trainSetNumber)))
    accuracyTemp1=np.zeros((len(testParamRange),len(kParam)))
    accuracyTemp2=np.zeros((len(testParamRange),len(kParam)))
    accuracyTemp3=np.zeros((len(testParamRange),len(forestParam)))
    accuracyTemp4=np.zeros((len(testParamRange),len(forestParam)))
    for testSetNumber in testParamRange:
        testDataPath = dataPath + '\\TestLatch2'
        
        #Import Data
        trainData, diagTrain1,diagTrainScale1,trainClass, featureChoice, xlabel,ylabel= diag.preprocessing(dataPath = trainDataPath, dataIndice = trainSetNumber,dataChoice = dataChoice, dataName = 'trainSet',diagDataChoice = diagDataChoice,plotFeatures = plotFeatures,save=save)
        testData, diagTest1 ,diagTestScale1,testClass,featureChoice, xlabel,ylabel= diag.preprocessing(dataPath = testDataPath, dataIndice = testSetNumber,dataChoice = dataChoice, dataName = 'testSet',diagDataChoice = diagDataChoice,plotFeatures = plotFeatures,save=save)
        featureName = featureChoice

        # We do a classification for all algorithms possible in order to compare them
        classifierChoice = 'Knn'
        i=0
        for k in kParam:
            trainClassifi1,trainRoc1,TrainCm1,trainCmAcc1 = diag.classifier(diagTrainScale1,trainClass,classifierChoice,knn_n_neighbors=k,figName='Train_Classif_'+featureChoice,plot=plotClassification,classesName=className,xlabel=xlabel,ylabel=ylabel,save=save,folderPath=savePath)
            accuracyPerPoint,accuracyRupture = diag.doResultClassification(testData,diagTestScale1,testClass,trainClassifi1,plot=0,xlabel='X',ylabel='Y',folderPath='',save=0,figName='result' + classifierChoice,savePath='',classifierChoice=classifierChoice,featureChoice=featureChoice)
            accuracyTemp1[testSetNumber-1,i] = accuracyPerPoint
            accuracyTemp2[testSetNumber-1,i] = accuracyRupture
            i=i+1
            
                # We do a classification for all algorithms possible in order to compare them
        classifierChoice = 'random_forest_classifier'
        i=0
        for ensemble in forestParam:
            trainClassifi1,trainRoc1,TrainCm1,trainCmAcc1 = diag.classifier(diagTrainScale1,trainClass,classifierChoice,ensemble_estimators=ensemble,figName='Train_Classif_'+featureChoice,plot=plotClassification,classesName=className,xlabel=xlabel,ylabel=ylabel,save=save,folderPath=savePath)
            accuracyPerPoint,accuracyRupture = diag.doResultClassification(testData,diagTestScale1,testClass,trainClassifi1,plot=0,xlabel='X',ylabel='Y',folderPath='',save=0,figName='result' + classifierChoice,savePath='',classifierChoice=classifierChoice,featureChoice=featureChoice)
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
'''       
        


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
    ensemble = 25
    modelForest,trainRoc1,TrainCm1,trainCmAcc1 = diag.classifier(diagTrainScale1,trainClass,'random_forest_classifier',ensemble_estimators=ensemble,figName='Train_Classif_'+featureChoice,plot=plotClassification,classesName=className,xlabel=xlabel,ylabel=ylabel,save=save,folderPath=savePath)
    modelSVM,trainRoc1,TrainCm1,trainCmAcc1 = diag.classifier(diagTrainScale1,trainClass,'svm',figName='Train_Classif_'+featureChoice,plot=plotClassification,classesName=className,xlabel=xlabel,ylabel=ylabel,save=save,folderPath=savePath)
    
    
    for testSetNumber in testRange:
        
        #Import Data
        testData, diagTest1 ,diagTestScale1,testClass,featureChoice, xlabel,ylabel = diag.preprocessing(dataColumnChoice = 2, dataPath = testDataPath, windowSize=windowSize, dataIndice = testSetNumber,dataChoice = dataChoice, dataName = ' ',diagDataChoice = diagDataChoice,plotFeatures = plotFeatures,save=save)
        featureName = featureChoice
        faultySetScale = scaler.transform(testData[:,2].reshape(-1,1))
        normalSet = scaler.transform(testData[:,1].reshape(-1,1))
        diagNormalScale = diag.statExtraction(normalSet,windowSize,diagDataChoice)
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

#Accuracy per Algorithm
'''
accuracyPerPointTotKnn = np.mean(accuracyPerPointKnn)
accuracyRuptTotKnn =  np.mean(accuracyRuptureKnn)
accuracyPerPointTotNB = np.mean(accuracyPerPointNB)
accuracyRuptTotNB = np.mean(accuracyRuptureNB)
accuracyPerPointTotDT = np.mean(accuracyPerPointDT)
accuracyRuptTotDT = np.mean(accuracyRuptureDT)
accuracyPerPointTotRF = np.mean(accuracyPerPointRF)
accuracyRuptTotRF = np.mean(accuracyRuptureRF)
accuracyPerPointTotSvm = np.mean(accuracyPerPointSvm)
accuracyRuptTotSvm = np.mean(accuracyRuptureSvm)


#Plot accuracy tot
PerPoint = [accuracyPerPointTotKnn,accuracyPerPointTotNB,accuracyPerPointTotDT,accuracyPerPointTotRF,accuracyPerPointTotSvm]
Rupt = [accuracyRuptTotKnn,accuracyRuptTotNB,accuracyRuptTotDT,accuracyRuptTotRF,accuracyRuptTotSvm]
fig1,axs = plt.subplots()
xAxis = range(0,len(classificationList))
plt.plot(xAxis,PerPoint, "r", label="Time window");
plt.plot(xAxis,Rupt, "b", label="Rupture algorithm");
plt.title('Classification accuracy for '+ str(trainSetNumber)+ ' train sets on ' + str(testSetNumber)+ ' test sets ');
plt.xlabel('Classification method')
plt.ylabel('Accuracy')
plt.ylim(ymax=1)
plt.grid();
plt.legend();
axs.set_xticks(range(0,len(classificationList)));
axs.set_xticklabels(classificationName);
diag.saveFigure(save,savePath,'/Classification_accuracy_'+ str(testSetNumber)+ '_test_sets.png');
plt.show()


Results1 = accuracyParamKnnPerPoint
Results2 = accuracyParamKnnRupt
Results3 = accuracyParamForestPerPoint
Results4 = accuracyParamForestRupt
Results5 = np.array([accuracyPerPointKnn,accuracyPerPointNB,accuracyPerPointDT,accuracyPerPointRF,accuracyPerPointSvm])
Results6 = np.array([accuracyPerPointTotKnn,accuracyPerPointTotNB,accuracyPerPointTotDT,accuracyPerPointTotRF,accuracyPerPointTotSvm])
Results7 = np.array([accuracyRuptureKnn,accuracyRuptureNB,accuracyRuptureDT,accuracyRuptureRF,accuracyRuptureSvm])
Results8 = np.array([accuracyRuptTotKnn,accuracyRuptTotNB,accuracyRuptTotDT,accuracyRuptTotRF,accuracyRuptTotSvm])

if saveResult:
    os.makedirs(resultPath, exist_ok=True)
    np.savetxt(resultPath + '\\accuracyParamKnnPerPoint.csv',Results1,delimiter=',')
    np.savetxt(resultPath + '\\accuracyParamKnnRupt.csv',Results2,delimiter=',')
    np.savetxt(resultPath + '\\accuracyParamForestPerPoint.csv',Results3,delimiter=',')
    np.savetxt(resultPath + '\\accuracyParamForestRupt.csv',Results4,delimiter=',')
    
    np.savetxt(resultPath + '\\accuracyPerPointKnn.csv',accuracyPerPointKnn,delimiter=',')
    np.savetxt(resultPath + '\\accuracyPerPointNB.csv',accuracyPerPointNB,delimiter=',')
    np.savetxt(resultPath + '\\accuracyPerPointDT.csv',accuracyPerPointDT,delimiter=',')
    np.savetxt(resultPath + '\\accuracyPerPointRF.csv',accuracyPerPointRF,delimiter=',')
    np.savetxt(resultPath + '\\accuracyPerPointSvm.csv',accuracyPerPointSvm,delimiter=',')
    
    np.savetxt(resultPath + '\\accuracyPerPointTot.csv',Results6,delimiter=',')
    
    np.savetxt(resultPath + '\\accuracyRuptureKnn.csv',accuracyRuptureKnn,delimiter=',')
    np.savetxt(resultPath + '\\accuracyRuptureNB.csv',accuracyRuptureNB,delimiter=',')
    np.savetxt(resultPath + '\\accuracyRuptureDT.csv',accuracyRuptureDT,delimiter=',')
    np.savetxt(resultPath + '\\accuracyRuptureRF.csv',accuracyRuptureRF,delimiter=',')
    np.savetxt(resultPath + '\\accuracyRuptureSvm.csv',accuracyRuptureSvm,delimiter=',')
    
    np.savetxt(resultPath + '\\ruptureTot.csv',Results6,delimiter=',')


'''

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
print("TP FP FN TN")

print("\nKNN:")
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




