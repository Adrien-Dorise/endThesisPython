# This program is an example for the diagnostic library
# Author: Adrien Dorise
# Date: September 2020

import detectionToolbox as diag
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import sys
import matplotlib.pyplot as plt 
import time
import os




# %config InlineBackend.figure_format = 'retina'



# Timer class
import time
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
    parameter = 5 #Can be changed to modify number of labels modified in the sliding window
    
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
        labelArray[j] = 100
    
    return faultyDataSet, labelArray


#General parameters
# dataPath = 'H:\\DIAG_RAD\\Programs\\Diagnostic_python\\DiagnosticExample\\ExampleDataSets'
# savePath = 'H:\\DIAG_RAD\\Results\\Diagnostic\\Example\\example1' 
dataChoice = 1 #1 for simulated data: 2 for real datas
dataPath = 'H:\DIAG_RAD\DataSets\Simulation_Matlab\datasGenerator\DataExemple'
savePathFolder = 'H:\\DIAG_RAD\\Results\\IFAC_Safeprocess_2021\\multiple_testSets\\3classes\\mean_variance_trainSet'
resultPath = 'H:\\DIAG_RAD\\Results\\IFAC_Safeprocess_2021\\Accuracy\\classificationAllStats\\classi2'
saveResult = 0

diagDataChoice = 1 # 1 (mean & variance); 2 (mean & frequency); 3 (variance & frequency); 4 (mean & min & max & variance & skewness & kurtosis); 5 (mean & min & max & variance & skewness & kurtosis & freq); 6 all stats

addNewLatchClass = 0

watchDogThreshold = 78
classifierChoice = 'Knn' # 'Knn', 'svm', 'decision_tree_classifier, 'random_forest_classifier', 'naive_bayes'
clusteringChoice = 'Kmeans' # 'Kmeans', 'HC'
n_clusters = 4

trainParamRange =  range(1,1+1)
testParamRange = range(1,1+1)
trainRange = range(1,1+1)
testRange = range(1,1+1)

doAddPointPerPoint = 1
doAddPointRupt = 1


plotFeatures = 0
plotTimeRupt = 0
plotClassification = 0
plotDiagRupture = 0
plotDiagPerPoints = 0
plotAccuracyPerMethods = 0
plotAccuracyAllSets = 0

save = 0







data=pd.DataFrame()
classificationList = ['Knn', 'naive_bayes', 'decision_tree_classifier', 'random_forest_classifier','svm']
classificationName = ['Knn', 'Naive_Bayes', 'Decision tree', 'Random Forest','SVM']
colors = ['green','red']
className = ['Normal','','','','','latch','front de latch up']
if save == 1 or saveResult == 1:
    inp = input('The parameter save equal 1. Do you really wish to save the datas (might overwritte older files)?\nY or N ?\n')
    if (inp == 'N' or inp == 'n' or inp == 'no' or inp == 'No'):
        sys.exit()


#accuracyPerPoint1 is the accuracy per points for each method listed in classificationList
#accuracyPerPoint2 is the confusion matrix per points for each method listed in classificationList
#accuracyRupture gives the accuracy per point for each method
#accuracyXtotal gives total accuracy for all methods listed and for multiple training sets




#Classification

#Finding Best parameters for each algorithm

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
    trainDataPath = dataPath + '\\TrainLatch2'
    data=data.append(diag.ifacDataFrame('train'+str(trainSetNumber)))
    trainData, diagTrain1,diagTrainScale1,trainClass, featureChoice, xlabel,ylabel= diag.preprocessing(dataPath = trainDataPath, dataIndice = trainSetNumber,dataChoice = dataChoice, dataName = 'trainSet',diagDataChoice = diagDataChoice,plotFeatures = plotFeatures,save=save)

    
    k=9
    modelKnn,trainRoc1,TrainCm1,trainCmAcc1 = diag.classifier(diagTrainScale1,trainClass,'Knn',knn_n_neighbors=k,figName='Train_Classif_'+featureChoice,plot=plotClassification,classesName=className,xlabel=xlabel,ylabel=ylabel,save=save,folderPath=savePath)
    modelBayes,trainRoc1,TrainCm1,trainCmAcc1 = diag.classifier(diagTrainScale1,trainClass,'naive_bayes',figName='Train_Classif_'+featureChoice,plot=plotClassification,classesName=className,xlabel=xlabel,ylabel=ylabel,save=save,folderPath=savePath)
    modelTree,trainRoc1,TrainCm1,trainCmAcc1 = diag.classifier(diagTrainScale1,trainClass,'decision_tree_classifier',figName='Train_Classif_'+featureChoice,plot=plotClassification,classesName=className,xlabel=xlabel,ylabel=ylabel,save=save,folderPath=savePath)
    ensemble = 25
    modelForest,trainRoc1,TrainCm1,trainCmAcc1 = diag.classifier(diagTrainScale1,trainClass,'random_forest_classifier',ensemble_estimators=ensemble,figName='Train_Classif_'+featureChoice,plot=plotClassification,classesName=className,xlabel=xlabel,ylabel=ylabel,save=save,folderPath=savePath)
    modelSVM,trainRoc1,TrainCm1,trainCmAcc1 = diag.classifier(diagTrainScale1,trainClass,'svm',figName='Train_Classif_'+featureChoice,plot=plotClassification,classesName=className,xlabel=xlabel,ylabel=ylabel,save=save,folderPath=savePath)

    
    for testSetNumber in testRange:
        testDataPath = dataPath + '\\TestLatch2'
        
        #Import Data
        testData, diagTest1 ,diagTestScale1,testClass,featureChoice, xlabel,ylabel= diag.preprocessing(dataPath = testDataPath, dataIndice = testSetNumber,dataChoice = dataChoice, dataName = 'testSet',diagDataChoice = diagDataChoice,plotFeatures = plotFeatures,save=save)
        featureName = featureChoice
        
        
       
        
        
        # We do a classification for all algorithms possible in order to compare them
        classifierChoice = 'Knn'
        with Timer() as timer: #PerPoint
            accuracyPerPoint,cmPerPoint = diag.doResultClassificationPerPoint(testData,diagTestScale1,testClass,modelKnn,plot=0,xlabel='X',ylabel='Y',folderPath='',save=0,figName='result' + classifierChoice,savePath='',classifierChoice=classifierChoice,featureChoice=featureChoice)
            accuracyPerPointKnn[testSetNumber-1,trainSetNumber-1] = accuracyPerPoint
            for j in range(len(cmPerPoint)):
                cmPerPointKnn[j] += cmPerPoint[j]
        timePerPointKnn.append(timer.interval)
        
        with Timer() as timer: #Rupture
            accuracyRupture,cmRupture = diag.doResultClassificationRupture(testData,diagTestScale1,testClass,modelKnn,plot=0,xlabel='X',ylabel='Y',folderPath='',save=0,figName='result' + classifierChoice,savePath='',classifierChoice=classifierChoice,featureChoice=featureChoice)
            accuracyRuptureKnn[testSetNumber-1,trainSetNumber-1] = accuracyRupture
            for j in range(len(cmRupture)):
                cmRuptKnn[j] += cmRupture[j] 
        timeRuptKnn.append(timer.interval)


        classifierChoice = 'naive_bayes'
        with Timer() as timer: #PerPoint
            accuracyPerPoint,cmPerPoint = diag.doResultClassificationPerPoint(testData,diagTestScale1,testClass,modelBayes,plot=0,xlabel='X',ylabel='Y',folderPath='',save=0,figName='result' + classifierChoice,savePath='',classifierChoice=classifierChoice,featureChoice=featureChoice)
            accuracyPerPointBayes[testSetNumber-1,trainSetNumber-1] = accuracyPerPoint
            for j in range(len(cmPerPoint)):
                cmPerPointBayes[j] += cmPerPoint[j]
        timePerPointBayes.append(timer.interval)
        
        with Timer() as timer: #Rupture
            accuracyRupture,cmRupture = diag.doResultClassificationRupture(testData,diagTestScale1,testClass,modelBayes,plot=0,xlabel='X',ylabel='Y',folderPath='',save=0,figName='result' + classifierChoice,savePath='',classifierChoice=classifierChoice,featureChoice=featureChoice)
            accuracyRuptureBayes[testSetNumber-1,trainSetNumber-1] = accuracyRupture
            for j in range(len(cmRupture)):
                cmRuptBayes[j] += cmRupture[j] 
        timeRuptBayes.append(timer.interval)


        classifierChoice = 'decision_tree_classifier'
        with Timer() as timer: #PerPoint
            accuracyPerPoint,cmPerPoint = diag.doResultClassificationPerPoint(testData,diagTestScale1,testClass,modelTree,plot=0,xlabel='X',ylabel='Y',folderPath='',save=0,figName='result' + classifierChoice,savePath='',classifierChoice=classifierChoice,featureChoice=featureChoice)
            accuracyPerPointTree[testSetNumber-1,trainSetNumber-1] = accuracyPerPoint
            for j in range(len(cmPerPoint)):
                cmPerPointTree[j] += cmPerPoint[j]
        timePerPointTree.append(timer.interval)
        
        with Timer() as timer: #Rupture
            accuracyRupture,cmRupture = diag.doResultClassificationRupture(testData,diagTestScale1,testClass,modelTree,plot=0,xlabel='X',ylabel='Y',folderPath='',save=0,figName='result' + classifierChoice,savePath='',classifierChoice=classifierChoice,featureChoice=featureChoice)
            accuracyRuptureTree[testSetNumber-1,trainSetNumber-1] = accuracyRupture
            for j in range(len(cmRupture)):
                cmRuptTree[j] += cmRupture[j] 
        timeRuptTree.append(timer.interval)
        
        
        classifierChoice = 'random_forest_classifier'
        with Timer() as timer: #PerPoint
            accuracyPerPoint,cmPerPoint = diag.doResultClassificationPerPoint(testData,diagTestScale1,testClass,modelForest,plot=0,xlabel='X',ylabel='Y',folderPath='',save=0,figName='result' + classifierChoice,savePath='',classifierChoice=classifierChoice,featureChoice=featureChoice)
            accuracyPerPointForest[testSetNumber-1,trainSetNumber-1] = accuracyPerPoint
            for j in range(len(cmPerPoint)):
                cmPerPointForest[j] += cmPerPoint[j]
        timePerPointForest.append(timer.interval)
        
        with Timer() as timer: #Rupture
            accuracyRupture,cmRupture = diag.doResultClassificationRupture(testData,diagTestScale1,testClass,modelForest,plot=0,xlabel='X',ylabel='Y',folderPath='',save=0,figName='result' + classifierChoice,savePath='',classifierChoice=classifierChoice,featureChoice=featureChoice)
            accuracyRuptureForest[testSetNumber-1,trainSetNumber-1] = accuracyRupture
            for j in range(len(cmRupture)):
                cmRuptForest[j] += cmRupture[j] 
        timeRuptForest.append(timer.interval)
        

        classifierChoice = 'svm'
        with Timer() as timer: #PerPoint
            accuracyPerPoint,cmPerPoint = diag.doResultClassificationPerPoint(testData,diagTestScale1,testClass,modelSVM,plot=0,xlabel='X',ylabel='Y',folderPath='',save=0,figName='result' + classifierChoice,savePath='',classifierChoice=classifierChoice,featureChoice=featureChoice)
            accuracyPerPointSVM[testSetNumber-1,trainSetNumber-1] = accuracyPerPoint
            for j in range(len(cmPerPoint)):
                cmPerPointSVM[j] += cmPerPoint[j]
        timePerPointSVM.append(timer.interval)
        
        with Timer() as timer: #Rupture
            accuracyRupture,cmRupture = diag.doResultClassificationRupture(testData,diagTestScale1,testClass,modelSVM,plot=0,xlabel='X',ylabel='Y',folderPath='',save=0,figName='result' + classifierChoice,savePath='',classifierChoice=classifierChoice,featureChoice=featureChoice)
            accuracyRuptureSVM[testSetNumber-1,trainSetNumber-1] = accuracyRupture
            for j in range(len(cmRupture)):
                cmRuptSVM[j] += cmRupture[j] 
        timeRuptSVM.append(timer.interval)
        
        
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
totalPointRupture = cmRuptSVM[0] + cmRuptSVM[1] + cmRuptSVM[2] + cmRuptSVM[3]

cmPercentPerPointKnn= [100*cmPerPointKnn[0]/totalPerPoint, 100*cmPerPointKnn[1]/totalPerPoint, 100*cmPerPointKnn[2]/totalPerPoint, 100*cmPerPointKnn[3]/totalPerPoint]
cmPercentPerPointBayes = [100*cmPerPointBayes[0]/totalPerPoint, 100*cmPerPointBayes[1]/totalPerPoint, 100*cmPerPointBayes[2]/totalPerPoint, 100*cmPerPointBayes[3]/totalPerPoint]
cmPercentPerPointTree = [100*cmPerPointTree[0]/totalPerPoint, 100*cmPerPointTree[1]/totalPerPoint, 100*cmPerPointTree[2]/totalPerPoint, 100*cmPerPointTree[3]/totalPerPoint]
cmPercentPerPointForest = [100*cmPerPointForest[0]/totalPerPoint, 100*cmPerPointForest[1]/totalPerPoint, 100*cmPerPointForest[2]/totalPerPoint, 100*cmPerPointForest[3]/totalPerPoint]
cmPercentPerPointSVM = [100*cmPerPointSVM[0]/totalPerPoint, 100*cmPerPointSVM[1]/totalPerPoint, 100*cmPerPointSVM[2]/totalPerPoint, 100*cmPerPointSVM[3]/totalPerPoint]


cmPercentRuptKnn= [100*cmRuptKnn[0]/totalPointRupture, 100*cmRuptKnn[1]/totalPointRupture, 100*cmRuptKnn[2]/totalPointRupture, 100*cmRuptKnn[3]/totalPointRupture]
cmPercentRuptBayes = [100*cmRuptBayes[0]/totalPointRupture, 100*cmRuptBayes[1]/totalPointRupture, 100*cmRuptBayes[2]/totalPointRupture, 100*cmRuptBayes[3]/totalPointRupture]
cmPercentRuptTree = [100*cmRuptTree[0]/totalPointRupture, 100*cmRuptTree[1]/totalPointRupture, 100*cmRuptTree[2]/totalPointRupture, 100*cmRuptTree[3]/totalPointRupture]
cmPercentRuptForest = [100*cmRuptForest[0]/totalPointRupture, 100*cmRuptForest[1]/totalPointRupture, 100*cmRuptForest[2]/totalPointRupture, 100*cmRuptForest[3]/totalPointRupture]
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

print("TP FP FN TN")
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




