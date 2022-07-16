# This program is an example for the diagnostic library
# Author: Adrien Dorise
# Date: September 2020

import detectionToolbox as diag
import deepToolbox as deep
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import sys
import matplotlib.pyplot as plt 
import time
import os

# import tensorflow as tf
# tf.enable_eager_execution()

# %config InlineBackend.figure_format = 'retina'


#Utilisation GPU
# import tensorflow as tf
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# tf.debugging.set_log_device_placement(True)

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
        

def doOCC(model, data, normalData, classes, windowSize, diagDataChoice, cm, time, plot):
    with Timer() as timer:
        faultyData = scaler.transform(data[:,2].reshape(-1,1))
        faultyData = diag.statExtraction(faultyData,windowSize,diagDataChoice)
        classTemp = classes.copy()
        for i in range(len(faultyData[:,0])):
            tempCM,tempPred = diag.confusionMatrixClassifier(faultyData[i,:],classTemp[i],model,1)
            # print('i: ' + str(i) + ' / pred: ' + str(tempPred[0]) + ' / class: ' + str(classTemp[i]))
            if(tempPred[0] == -1 and classTemp[i] == 1):
                faultyData, classTemp = anomalyRemoval(faultyData,normalData,classTemp,1,i,windowSize)
            cm = np.add(cm ,tempCM)
            predictionResult.append(tempPred[0])
    time.append(timer.interval)
    if(plotTest):
        diag.plotOCC(faultyData,testClassOCC,modelOCSVM,figName='OC_SVM',xlabel='Variance',ylabel='Current',save=save,folderPath=savePath)

    return cm ,time


            
'''
#One class classification tests (Code taken from Example folder)
plotStats = 1
plotOCC = 1
save=0


    #Import Data
trainDataPathTemp = 'H:\DIAG_RAD\DataSets\Simulation_Matlab\datasGenerator\DataExemple\\TrainLatch2\\datas1\\diagData.txt'
testDataPathTemp = 'H:\DIAG_RAD\DataSets\Simulation_Matlab\datasGenerator\DataExemple\\TestLatch2\\datas1\\diagData.txt'
savePath = 'H:\\DIAG_RAD\\Results\\IFAC_Safeprocess_2022\\OCC'

# trainDataPathTemp = '/media/adrien/Adrien_Dorise_USB/DIAG_RAD/DataSets/Simulation_Matlab/datasGenerator/DataExemple/TrainLatch2/datas1/diagData.txt'
# testDataPathTemp = '/media/adrien/Adrien_Dorise_USB/DIAG_RAD/DataSets/Simulation_Matlab/datasGenerator/DataExemple/TestLatch2/datas1/diagData.txt'
# savePath = '/media/adrien/Adrien_Dorise_USB/DIAG_RAD/Results/datasGenerator/IFAC_Safeprocess_2022/OCC'

separator = ','
trainData = diag.importTextData(trainDataPathTemp,3,separator)
diag.plotCurrent(trainData[:,0],trainData[:,1],trainData[:,2],save=save,folderPath=savePath,figName = 'trainSet_consumption_current')
testData = diag.importTextData(testDataPathTemp,3,separator)
diag.plotCurrent(testData[:,0],testData[:,1],testData[:,2],save=save,folderPath=savePath,figName = 'testSet_consumption_current')

    #Import Class
className = ['normal','calculation','I/O','temperature','reset','anomaly','front de latch up']
colors = ['green','red']
trainClassDataPathTemp = 'H:\DIAG_RAD\DataSets\Simulation_Matlab\datasGenerator\DataExemple\\TrainLatch2\\datas1\\statusData.txt'
testClassDataPathTemp = 'H:\DIAG_RAD\DataSets\Simulation_Matlab\datasGenerator\DataExemple\\TestLatch2\\datas1\\statusData.txt'
# trainClassDataPathTemp = '/media/adrien/Adrien_Dorise_USB/DIAG_RAD/DataSets/Simulation_Matlab/datasGenerator/DataExemple/TrainLatch2/datas1/statusData.txt'
# testClassDataPathTemp = '/media/adrien/Adrien_Dorise_USB/DIAG_RAD/DataSets/Simulation_Matlab/datasGenerator/DataExemple/TestLatch2/datas1/statusData.txt'
trainClass = pd.DataFrame(data=diag.importTextData(trainClassDataPathTemp,6,separator=',')[:,5],columns=['latch up']).iloc[:,0]
trainClass[np.where(trainClass == 5)[0]] = 0 #All data are normal for train set
testClass = pd.DataFrame(data=diag.importTextData(testClassDataPathTemp,6,separator=',')[:,5],columns=['latch up']).iloc[:,0]
testClass[np.where(testClass == 5)[0]] = 1 #Fault values of 5 switch to 1

    #Create features
statDataPerPoint = diag.statsPerPoint(trainData[:,1],save=save,folderPath=savePath,plot=0,figName='train_set_StatsPerPoint')
[trainMinPerPoint,trainMaxPerPoint,trainMeanPerPoint,trainVariancePerPoint,trainSkewnessPerPoint,trainKurtosisPerPoint] = [statDataPerPoint[:,1],statDataPerPoint[:,2],statDataPerPoint[:,3],statDataPerPoint[:,4],statDataPerPoint[:,5],statDataPerPoint[:,6]]
statDataPerPoint = diag.statsPerPoint(testData[:,2],plot=0,save=save,folderPath=savePath,figName='test_set_StatsPerPoint')
[testMinPerPoint,testMaxPerPoint,testMeanPerPoint,testVariancePerPoint,testSkewnessPerPoint,testKurtosisPerPoint] = [statDataPerPoint[:,1],statDataPerPoint[:,2],statDataPerPoint[:,3],statDataPerPoint[:,4],statDataPerPoint[:,5],statDataPerPoint[:,6]]

    #Diag data creation
diagTrain1 =  np.array([trainVariancePerPoint,trainMeanPerPoint]).transpose()
diagTest1 =  np.array([testVariancePerPoint,testMeanPerPoint]).transpose()

    #Scaling
sc1 = StandardScaler();
diagTrainScale1 = sc1.fit_transform(diagTrain1);  
diagTestScale1 = sc1.transform(diagTest1)
    
#    Plotting of features
if plotStats:
    diag.plotLabelPoints(diagTrainScale1, trainClass, className,figName='Train_Scaled_Stats',colors=colors,xlabel='Variance',ylabel='Current',save=save,folderPath=savePath)
    diag.plotLabelPoints(diagTestScale1, testClass, className,figName='Test_Scaled_Stats',colors=colors,xlabel='Variance',ylabel='Current',save=save,folderPath=savePath)

    #Training
trainClassifi1 = diag.oneClassClassification(diagTrainScale1,classifierChoice='OCSVM',withPoints=1,figName='OC_SVM',plot=plotOCC,xlabel='Variance',ylabel='Current',save=save,folderPath=savePath)

    #Testing
_,a=diag.plotOCC(diagTestScale1,testClass,trainClassifi1,figName='Train_Classif_Stats',xlabel='Variance',ylabel='Current',save=save,folderPath=savePath)
confusionMatrix = [0,0,0,0]
prediction = []
testClassOCC = testClass
testClassOCC[np.where(testClassOCC == 0)[0]] = False
testClassOCC[np.where(testClassOCC == 1)[0]] = True
for i in range(len(diagTestScale1)):
    tempCM = diag.confusionMatrixOCC(diagTestScale1[i,:],testClassOCC[i],trainClassifi1)
    confusionMatrix = np.add(confusionMatrix,tempCM)
    prediction.append(trainClassifi1.predict(diagTestScale1[i,:].reshape(1,-1)))
print(confusionMatrix)

    #Other OCC test
diag.oneClassClassification(diagTrainScale1,classifierChoice='OCSVM',figName='OCSVM',plot=plotOCC,xlabel='Variance',ylabel='Current',save=save,folderPath=savePath)
diag.oneClassClassification(diagTrainScale1,classifierChoice='elliptic classification',figName='Elliptic_classification',plot=plotOCC,xlabel='Variance',ylabel='Current',save=save,folderPath=savePath)
diag.oneClassClassification(diagTrainScale1,classifierChoice='LOF',figName='LOF',plot=plotOCC,xlabel='Variance',ylabel='Current',save=save,folderPath=savePath)
diag.oneClassClassification(diagTrainScale1,classifierChoice='isolation forest',figName='Isolation_forest',plot=plotOCC,xlabel='Variance',ylabel='Current',save=save,folderPath=savePath)
   

    #Auto Encoders
model,threshold = deep.trainAE(diagTrainScale1,epochs = 250,batch_size=216)
prediction = deep.predictAE(model,threshold,diagTestScale1)
cmAE = deep.confusionMatrixAE(prediction, testClassOCC)

'''

#Confusion matrix of algorithms for multiple data sets

# dataPath = 'H:\\DIAG_RAD\\Programs\\Diagnostic_python\\DiagnosticExample\\ExampleDataSets'
# savePath = 'H:\\DIAG_RAD\\Results\\Diagnostic\\Example\\example1' 
dataChoice = 1 #1 for simulated data: 2 for real datas
trainDataPath = "H:\\DIAG_RAD\\DataSets\\\endThesisValidationData\\simulations\\trainSet\\microLatch"
testDataPath = "H:\\DIAG_RAD\\DataSets\\\endThesisValidationData\\simulations\\testSet\\microLatch"
# dataPath = '/media/adrien/Adrien_Dorise_USB1/DIAG_RAD/DataSets/Simulation_Matlab/datasGenerator/DataExemple'
savePathFolder = 'H:\\DIAG_RAD\\Results\\IFAC_Safeprocess_2022\\multiple_testSets\\3classes\\mean_variance_trainSet'
resultPath = 'H:\\DIAG_RAD\\Results\\IFAC_Safeprocess_2022\\Accuracy\\classificationAllStats\\classi2'
saveResult = 0

bigLatch = False #Adjust the labels in the time window
diagDataChoice = 1 # 1 (mean & variance); 2 (mean & frequency); 3 (variance & frequency); 4 (mean & min & max & variance & skewness & kurtosis); 5 (mean & min & max & variance & skewness & kurtosis & freq); 6 all stats
windowSize = 20

trainRange = range(1,1+1)
testRange = range(1,1+1)

plotTrain=0
plotTest=0
plotFeatures=0
save = 0


predictionResult = []
classCount= [0,0]

data=pd.DataFrame()
classificationList = ['OCSVM', 'elliptic classification', 'LOF', 'isolation forest']
classificationName = ['OCSVM', 'Elliptic classification', 'LOF', 'Isolation forest']
colors = ['green','red']
className = ['normal','','','','','latch','front de latch up']
if save == 1 or saveResult == 1:
    inp = input('The parameter save equal 1. Do you really wish to save the datas (might overwritte older files)?\nY or N ?\n')
    if (inp == 'N' or inp == 'n' or inp == 'no' or inp == 'No'):
        sys.exit()

scaler = MinMaxScaler(feature_range=(0,1))
cmOCSVM, cmEC, cmLOF, cmIF, cmAE = [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0] #TP FP FN TN
timeOCSVM,timeEC,timeLOF,timeIF,timeAE = [],[],[],[],[]
for trainSetNumber in trainRange:
    savePath = savePathFolder + str(trainSetNumber)
    data=data.append(diag.ifacDataFrame('train'+str(trainSetNumber)))
    trainData, diagTrain1,diagTrainScale1,trainClass, featureChoice, xlabel,ylabel= diag.preprocessing(dataPath = trainDataPath, windowSize=windowSize, dataIndice = trainSetNumber,dataChoice = dataChoice, dataColumnChoice=1, dataName = 'trainSet',diagDataChoice = diagDataChoice,plotFeatures = plotFeatures,save=save)
    # trainClass[np.where(trainClass == 5)[0]] = 0 
    trainScale = scaler.fit_transform(trainData[:,1].reshape(-1,1))
    diagTrainScale1 = diag.statExtraction(trainScale,windowSize,diagDataChoice)
    if(plotFeatures):
        diag.plotLabelPoints(diagTrainScale1, trainClass, className,figName='trainSet',colors=colors,xlabel=xlabel,ylabel=ylabel,save=save,folderPath=savePath)

    

    #training
    modelAE,thresholdAE = deep.trainAE(diagTrainScale1,epochs = 50,batch_size=216)
    classifierChoice = 'OCSVM'
    modelOCSVM = diag.oneClassClassification(diagTrainScale1,classifierChoice=classifierChoice,figName='OC_SVM',plot=plotTrain,xlabel='Variance',ylabel='Current',save=save,folderPath=savePath)
    classifierChoice = 'elliptic classification'
    modelEC = diag.oneClassClassification(diagTrainScale1,classifierChoice=classifierChoice,figName='Elliptic_classification',plot=plotTrain,xlabel='Variance',ylabel='Current',save=save,folderPath=savePath)
    classifierChoice = 'LOF'
    modelLOF = diag.oneClassClassification(diagTrainScale1,classifierChoice=classifierChoice,figName='LOF',plot=plotTrain,xlabel='Variance',ylabel='Current',save=save,folderPath=savePath)
    classifierChoice = 'isolation forest'
    modelIF = diag.oneClassClassification(diagTrainScale1,classifierChoice=classifierChoice,figName='Isolation_forest',plot=plotTrain,xlabel='Variance',ylabel='Current',save=save,folderPath=savePath)


    #testing
    for testSetNumber in testRange:
        
        #Import Data
        testData, diagTest1 ,diagTestScale1,testClass,featureChoice, xlabel,ylabel = diag.preprocessing(dataColumnChoice = 2, dataPath = testDataPath, windowSize=windowSize, dataIndice = testSetNumber,dataChoice = dataChoice, dataName = ' ',diagDataChoice = diagDataChoice,plotFeatures = plotFeatures,save=save)
        featureName = featureChoice
        faultySetScale = scaler.transform(testData[:,2].reshape(-1,1))
        normalSet = scaler.transform(testData[:,1].reshape(-1,1))
        diagNormalScale = diag.statExtraction(normalSet,windowSize,diagDataChoice)
        testClassOCC = testClass.copy()
        testClassOCC[np.where(testClassOCC != 5)[0]] = 0
        testClassOCC[np.where(testClassOCC == 5)[0]] = 1
        
        
        #!!!!!!This function is to be used when the fault value is too important and mess up witht the sliding window!!!!!!!!!!!!!!!!!!!!!
        if bigLatch:
            testClassOCC = getLabel(testClassOCC,1,windowSize)
        
    

        classifierChoice = 'OCSVM'
        cmOCSVM, timeOCSVM = doOCC(modelOCSVM, testData, diagNormalScale, testClassOCC, windowSize, diagDataChoice, cmOCSVM, timeOCSVM, plotTest)
        

        classifierChoice = 'elliptic classification'
        cmEC, timeEC = doOCC(modelEC, testData, diagNormalScale, testClassOCC, windowSize, diagDataChoice, cmEC, timeEC, plotTest)


        classifierChoice = 'LOF'
        cmLOF, timeLOF = doOCC(modelLOF, testData, diagNormalScale, testClassOCC, windowSize, diagDataChoice, cmLOF, timeLOF, plotTest)

      

        classifierChoice = 'isolation forest'
        cmIF, timeIF = doOCC(modelIF, testData, diagNormalScale, testClassOCC, windowSize, diagDataChoice, cmIF, timeIF, plotTest)


        #Auto encoder
        with Timer() as timer:
            faultyDataAE = scaler.transform(testData[:,2].reshape(-1,1))
            faultyDataAE = diag.statExtraction(faultyDataAE,windowSize,diagDataChoice)
            classAE = testClassOCC.copy()
            
            # for i in range(len(diagTestScale1)):
            #       tempPred = deep.predictAE(modelAE,thresholdAE,faultyDataAE[i,:])
            #       tempCMAE = deep.confusionMatrixAE(tempPred, classAE[i],1)
            #       if(tempPred[0] == 0 and classAE[i] == 1):
            #         faultyDataAE, classAE = anomalyRemoval(faultyDataAE,diagNormalScale,classAE,1,i,windowSize)
            #       cmAE = np.add(cmAE,tempCMAE)
            
            predictionAE = deep.predictAE(modelAE,thresholdAE,faultyDataAE)
            tempCMAE = deep.confusionMatrixAE(predictionAE, classAE,1)
            cmAE = np.add(cmAE,tempCMAE)
            
        timeAE.append(timer.interval)

        
        print('\nACCURACY for trainSet ' + str(trainSetNumber) + ' in test set ' + str(testSetNumber) + '\n')




totalPoint = cmOCSVM[0] + cmOCSVM[1] + cmOCSVM[2] + cmOCSVM[3]
cmPercentOVSVM = [100*cmOCSVM[0]/totalPoint, 100*cmOCSVM[1]/totalPoint, 100*cmOCSVM[2]/totalPoint, 100*cmOCSVM[3]/totalPoint]
cmPercentEC = [100*cmEC[0]/totalPoint, 100*cmEC[1]/totalPoint, 100*cmEC[2]/totalPoint, 100*cmEC[3]/totalPoint]
cmPercentLOF = [100*cmLOF[0]/totalPoint, 100*cmLOF[1]/totalPoint, 100*cmLOF[2]/totalPoint, 100*cmLOF[3]/totalPoint]
cmPercentIF = [100*cmIF[0]/totalPoint, 100*cmIF[1]/totalPoint, 100*cmIF[2]/totalPoint, 100*cmIF[3]/totalPoint]
cmPercentAE = [100*cmAE[0]/totalPoint, 100*cmAE[1]/totalPoint, 100*cmAE[2]/totalPoint, 100*cmAE[3]/totalPoint]

meanTimeOCSVM = sum(timeOCSVM)/len(timeOCSVM)
meanTimeEC = sum(timeEC)/len(timeEC)
meanTimeLOF = sum(timeLOF)/len(timeLOF)
meanTimeIF = sum(timeIF)/len(timeIF)
meanTimeAE = sum(timeAE)/len(timeAE)
print("TP FP FN TN")
print("Train set: " + trainDataPath)
print("Test set: " + testDataPath)
print("TP FP FN TN")
print("OCSVM:")
print(cmOCSVM)
print(cmPercentOVSVM)
print("mean time: " + str(meanTimeOCSVM))
print("EC:")
print(cmEC)
print(cmPercentEC)
print("mean time: " + str(meanTimeEC))
print("LOF:")
print(cmLOF)
print(cmPercentLOF)
print("mean time: " + str(meanTimeLOF))
print("IF:")
print(cmIF)
print(cmPercentIF)
print("mean time: " + str(meanTimeIF))
print("AE:")
print(cmAE)
print(cmPercentAE)
print("mean time: " + str(meanTimeAE))



'''
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






