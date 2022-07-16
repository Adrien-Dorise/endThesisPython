import detectionToolbox as diag
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import sys
import matplotlib.pyplot as plt 
from sklearn import preprocessing
import os
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


def modifyDyclee(g_size):
    file = open('H:\\DIAG_RAD\\Programs\\Dyclee\\2020_11\\bin\\file.txt','r')
    DycleeFile = file.read()
    file.close()
    DycleeFile.find('g_size = ')
    DycleeFile=DycleeFile.replace(DycleeFile[19:24],g_size)
    file = open('H:\\DIAG_RAD\\Programs\\Dyclee\\2020_11\\bin\\file.txt','w')
    file.write(DycleeFile)
    file.close()


def doClustering(model,data):
    return 1

def doClassifPerPoint(model, data, classes, faultValue, cm, time, plot):
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


def doClassifRupture(model, data, timeSeries, classes, faultValue, rupturePenalty, cm, time, plot) :
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






# 'H:\\DIAG_RAD\\Programs\\Diagnostic_python\\DiagnosticExample\\ExampleDataSets\\AllDefectAdded.txt'
# 'H:\\DIAG_RAD\\Programs\\Diagnostic_python\\DiagnosticExample\\ExampleDataSets\\All16.txt'

dataChoice = 1 #1 for simulated data: 2 for real datas
trainDataPath = "H:\\DIAG_RAD\\DataSets\\\endThesisValidationData\\simulations\\trainSet\\DestructiveLatch"
testDataPath = "H:\\DIAG_RAD\\DataSets\\\endThesisValidationData\\simulations\\testSet\\DestructiveLatch"
savePathFolder = 'H:\\DIAG_RAD\\Results\\IFAC_Safeprocess_2021\\multiple_testSets\\3classes\\mean_variance_trainSet'
resultPath = 'H:\\DIAG_RAD\\Results\\IFAC_Safeprocess_2021\\Accuracy\\clusteringAllStats\\test4'

diagDataChoice = 1 # 1 (mean & variance); 2 (mean & frequency); 3 (variance & frequency); 4 (mean & min & max & variance & skewness & kurtosis); 5 (mean & min & max & variance & skewness & kurtosis & freq)

addNewLatchClass = 0
bigLatch = False #Adjust the labels in the time window
windowSize = 10
rupturePenalty = 0.8
faultValue = 1
classifModelSelection = "svm"

dataName = 'testSet'
trainRange = range(1,1+1)
testRange = range(1,1+1)
dataParamRange =  range(1,1+1)


penaltyValue = 25
ratioPenalty = 10

saveResult = 0
save=0


savePath = 'H:\\DIAG_RAD\\Results\\IFAC_Safeprocess_2021\\multiple_testSets\\3classes\\mean_variance_trainSet'

plotClustering = 1
plotParallelFeatures = 0

plotFeatures = 0
plotClassification = 0
plotDiagRupture = 0
plotDiagPerPoints = 0

plotTrain=0
plotTest=0




#Init some parameters
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



# Clustering
'''
#Finding Best parameters for each algorithm
KmeansNClust = [1,2,3,4,5,6]
HCmetric = ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']
# DBeps = [0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7]
DBeps = [0.0005,0.0008,0.001,0.003,0.005,0.008,0.01,0.03,0.05,0.1]
DBmetric = ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']
Dycleegsize = ['0.040','0.050','0.060','0.070','0.080','0.100','0.120','0.140','0.160','0.180','0.200','0.250']
# Dycleegsize = ['0.100','0.120','0.140','0.160','0.180','0.200','0.220','0.240','0.260']
accuracyParamKmeans = np.zeros((len(dataParamRange),len(KmeansNClust)))
accuracyParamHC = np.zeros((len(dataParamRange),len(HCmetric)))
accuracyParamDB = np.zeros((len(dataParamRange),len(DBeps)))
accuracyParamDB2 = np.zeros((len(dataParamRange),len(DBmetric)))
accuracyParamDyclee = np.zeros((len(dataParamRange),len(Dycleegsize)))

scoreParamKmeans = np.zeros((len(dataParamRange),len(KmeansNClust)))
scoreParamHC = np.zeros((len(dataParamRange),len(KmeansNClust)))
scoreParamDB = np.zeros((len(dataParamRange),len(DBeps)))
scoreParamDB2 = np.zeros((len(dataParamRange),len(DBmetric)))
scoreParamDyclee = np.zeros((len(dataParamRange),len(Dycleegsize)))
#Finding Best parameters for each algorithm
i=0
j=0
for indice in dataParamRange:
    dataIndice = indice
    print('Indice number ' + str(indice))
    timeSerie, features,dataClustering,classClustering, featureChoice, xlabel,ylabel= diag.preprocessing(dataPath = dataPath, dataIndice = dataIndice, dataName = dataName,diagDataChoice = diagDataChoice)
    
    
    clusteringChoice = 'Kmeans'
    testCluster1,testResult1,testPredict1 = diag.clustering(dataClustering,clusteringChoice,epsilon= epsilon,n_clusters=n_clusters,figName='Test_Clust_'+featureChoice,xlabel=xlabel,ylabel=ylabel,save=save,folderPath=savePath,plot=plotClustering,doEvaluation = 0)
    accuracyCluster1,clusterClass1 = diag.doResultClustering(timeSerie,testCluster1.labels_,classClustering,figName = clusteringChoice)
    accuracyParamKmeans[i,j] = accuracyCluster1
    
    clusteringChoice = 'HC'
    for metr in  HCmetric:
        metric=metr
        testCluster1,testResult1,testPredict1 = diag.clustering(dataClustering,clusteringChoice,epsilon= epsilon,n_clusters=n_clusters,metric=metric,figName='Test_Clust_'+featureChoice,xlabel=xlabel,ylabel=ylabel,save=save,folderPath=savePath,plot=plotClustering,doEvaluation = 0)
        accuracyCluster2,clusterClass2 = diag.doResultClustering(timeSerie,testCluster1.labels_,classClustering,figName = clusteringChoice)
        accuracyParamHC[i,j] = accuracyCluster2
        scoreParamHC[i,j] = diag.clusterScore(accuracyParamHC[i,j],len(np.unique(classClustering)),len(np.unique(testCluster1.labels_)),ratioPenalty,penaltyValue)
        j=j+1
    j=0
    metric='cosine'
    
    clusteringChoice = 'DBSCAN'
    for eps in  DBeps:
        epsilon = eps
        testCluster1,testResult1,testPredict1 = diag.clustering(dataClustering,clusteringChoice,epsilon= epsilon,n_clusters=n_clusters,metric=metric,figName='Test_Clust_'+featureChoice,xlabel=xlabel,ylabel=ylabel,save=save,folderPath=savePath,plot=plotClustering,doEvaluation = 0)
        print('epsilon is ' + str(eps))
        accuracyCluster3,clusterClass3 = diag.doResultClustering(timeSerie,testCluster1.labels_,classClustering,figName = clusteringChoice)
        accuracyParamDB[i,j] = accuracyCluster3
        scoreParamDB[i,j] = diag.clusterScore(accuracyParamDB[i,j],len(np.unique(classClustering)),len(np.unique(testCluster1.labels_)),ratioPenalty,penaltyValue)
        j=j+1
    j=0
    epsilon = 0.08

    for metr in DBmetric :
        metric=metr
        testCluster1,testResult1,testPredict1 = diag.clustering(dataClustering,clusteringChoice,epsilon= epsilon,n_clusters=n_clusters,metric=metric,figName='Test_Clust_'+featureChoice,xlabel=xlabel,ylabel=ylabel,save=save,folderPath=savePath,plot=plotClustering,doEvaluation = 0)
        print(print('metric is ' + str(metr)))
        accuracyCluster3,clusterClass3 = diag.doResultClustering(timeSerie,testCluster1.labels_,classClustering,figName = clusteringChoice)
        accuracyParamDB2[i,j] = accuracyCluster3
        scoreParamDB2[i,j] = diag.clusterScore(accuracyParamDB2[i,j],len(np.unique(classClustering)),len(np.unique(testCluster1.labels_)),ratioPenalty,penaltyValue)
        j=j+1
    j=0
    metric='cosine'
    
    #DyClee
    for gSiz in Dycleegsize:
        #Call Dyclee
        g_size = gSiz
        print('\nDyClee in progress...')
        modifyDyclee(g_size)
        scaler = preprocessing.MinMaxScaler()
        normalized_X = scaler.fit_transform(dataClustering)
        np.savetxt('H:\\DIAG_RAD\\Programs\\Dyclee\\2020_11\\data\\data.dat',normalized_X,delimiter=',')
        os.system('H:\\DIAG_RAD\\Programs\\Dyclee\\2020_11\\bin\\DyClee H:\\DIAG_RAD\\Programs\\Dyclee\\2020_11\\bin\\file.txt')
        #Use Dyclee class in Python
        DycleeClass = np.genfromtxt('H:\\DIAG_RAD\\Results\\Diagnostic\\DyClee\\DycleeResult.dat',dtype=int)
        DycleeClass = np.subtract(DycleeClass, np.ones(len(DycleeClass),dtype=int))
        while len(DycleeClass) < len(classClustering):
            DycleeClass = np.append(DycleeClass,DycleeClass[len(DycleeClass)-1])
        accuracyCluster4,clusterClass4 = diag.doResultClustering(timeSerie,DycleeClass,classClustering,figName = 'DyClee')
        accuracyParamDyclee[i,j] = accuracyCluster4
        scoreParamDyclee[i,j] = diag.clusterScore(accuracyParamDyclee[i,j],len(np.unique(classClustering)),len(np.unique(DycleeClass)),ratioPenalty,penaltyValue)
        print('DyClee done !\n')
        j=j+1
    j=0    
    i=i+1

i=0

'''


#Accuracy calculation with classification algorithm used
accuracyKmeans,accuracyHC,accuracyDB,accuracyDyclee = [],[],[],[]
accuracyClassifKmeans,accuracyClassifHC,accuracyClassifDB,accuracyClassifDyclee = [],[],[],[]
scoreDB,scoreDyclee = [],[]
scoreClassifDB,scoreClassifDyclee,scoreClassifRuptDB,scoreClassifRuptDyclee = [],[],[],[]
accuracyClassifRuptKmeans,accuracyClassifRuptHC,accuracyClassifRuptDB,accuracyClassifRuptDyclee = [],[],[],[]

cmPerPointKmeans,cmRuptKmeans = [0,0,0,0], [0,0,0,0] #TP,FP,FN,TN
cmPerPointHC,cmRuptHC = [0,0,0,0], [0,0,0,0] #TP,FP,FN,TN
cmPerPointDBSCAN,cmRuptDBSCAN = [0,0,0,0], [0,0,0,0] #TP,FP,FN,TN
cmPerPointDyClee,cmRuptDyClee = [0,0,0,0], [0,0,0,0] #TP,FP,FN,TN

timePerPointKmeans,timeRuptKmeans = [], [] 
timePerPointHC,timeRuptHC = [], [] 
timePerPointDBSCAN,timeRuptDBSCAN = [], []
timePerPointDyClee,timeRuptDyClee = [], []


i=0
j=0
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
        
    n_clusters = 3
    modelClustKmeans,testKmeans,testKmeans = diag.clustering(diagTrainScale1,'Kmeans',n_clusters=n_clusters,figName='Test_Clust_'+featureChoice,xlabel=xlabel,ylabel=ylabel,save=save,folderPath=savePath,plot=plotClustering,doEvaluation = 0)
    accuracyKmeans,classKmeans = diag.clustertingExpertOpinion(trainData,modelClustKmeans.labels_,trainClass,figName = 'Kmeans')
    modelClassifKmeans,trainRoc1,TrainCm1,trainCmAcc1 = diag.classifier(diagTrainScale1,classKmeans,classifModelSelection)
    
    n_clusters = 3
    modelClustHC,testHC,testHC = diag.clustering(diagTrainScale1,'HC',n_clusters=n_clusters,figName='Test_Clust_'+featureChoice,xlabel=xlabel,ylabel=ylabel,save=save,folderPath=savePath,plot=plotClustering,doEvaluation = 0)
    accuracyHC,classHC = diag.clustertingExpertOpinion(trainData,modelClustHC.labels_,trainClass,figName = 'HC')
    modelClassifHC,trainRoc1,TrainCm1,trainCmAcc1 = diag.classifier(diagTrainScale1,classHC,classifModelSelection)
    
    epsilon = 0.08
    metric='cosine'
    modelClustDBSCAN,testDBSCAN,testDBSCAN = diag.clustering(diagTrainScale1,'DBSCAN',epsilon= epsilon,figName='Test_Clust_'+featureChoice,xlabel=xlabel,ylabel=ylabel,save=save,folderPath=savePath,plot=plotClustering,doEvaluation = 0)
    accuracyDBSCAN,classDBSCAN = diag.clustertingExpertOpinion(trainData,modelClustDBSCAN.labels_,trainClass,figName = 'DBSCAN')
    if len(np.unique(classDBSCAN)) <=1:
        classDBSCAN[0]=1
    modelClassifDBSCAN,trainRoc1,TrainCm1,trainCmAcc1 = diag.classifier(diagTrainScale1,classDBSCAN,classifModelSelection)
    
    #Call Dyclee
    print('\nDyClee in progress...')
    modifyDyclee('0.05')
    scalerDyClee = preprocessing.MinMaxScaler()
    normalized_X = scalerDyClee.fit_transform(diagTrainScale1)
    np.savetxt('H:\\DIAG_RAD\\Programs\\Dyclee\\2020_11\\data\\data.dat',normalized_X,delimiter=',')
    os.system('H:\\DIAG_RAD\\Programs\\Dyclee\\2020_11\\bin\\DyClee H:\\DIAG_RAD\\Programs\\Dyclee\\2020_11\\bin\\file.txt')
    #Use Dyclee class in Python
    DycleeClass = np.genfromtxt('H:\\DIAG_RAD\\Results\\Diagnostic\\DyClee\\DycleeResult.dat',dtype=int)
    DycleeClass = np.subtract(DycleeClass, np.ones(len(DycleeClass),dtype=int))
    while len(DycleeClass) < len(trainClass):
        DycleeClass = np.append(DycleeClass,DycleeClass[len(DycleeClass)-1])
    accuracyDyClee,classDyClee = diag.clustertingExpertOpinion(diagTrainScale1,DycleeClass,trainClass,figName = 'DyClee')
    if len(np.unique(classDyClee)) <=1:
        classDyClee[0]=1
    modelClassifDyClee,trainRoc1,TrainCm1,trainCmAcc1 = diag.classifier(diagTrainScale1,classDyClee,classifModelSelection)
    

    
    
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
    
    
    
    clusteringChoice = 'Kmeans'
    cmPerPointKmeans,timePerPointKmeans = doClassifPerPoint(modelClassifKmeans, diagTestScale1, testClassClassif, faultValue,cmPerPointKmeans,timePerPointKmeans, plotDiagPerPoints)
    cmRuptKmeans,timeRuptKmeans = doClassifRupture(modelClassifKmeans, diagTestScale1, testData, testClassClassif, faultValue, rupturePenalty, cmRuptKmeans, timeRuptKmeans, plotDiagPerPoints)
    
    
    
    clusteringChoice = 'HC'
    cmPerPointHC,timePerPointHC = doClassifPerPoint(modelClassifHC, diagTestScale1, testClassClassif, faultValue,cmPerPointHC,timePerPointHC, plotDiagPerPoints)
    cmRuptHC,timeRuptHC = doClassifRupture(modelClassifHC, diagTestScale1, testData, testClassClassif, faultValue, rupturePenalty, cmRuptHC, timeRuptHC, plotDiagPerPoints)
    
    
    
    clusteringChoice = 'DBSCAN'
    cmPerPointDBSCAN,timePerPointDBSCAN = doClassifPerPoint(modelClassifDBSCAN, diagTestScale1, testClassClassif, faultValue,cmPerPointDBSCAN,timePerPointDBSCAN, plotDiagPerPoints)
    cmRuptDBSCAN,timeRuptDBSCAN = doClassifRupture(modelClassifDBSCAN, diagTestScale1, testData, testClassClassif, faultValue, rupturePenalty, cmRuptDBSCAN, timeRuptDBSCAN, plotDiagPerPoints)
    
    
    # accuracyClassifDB.append(np.mean(accuracyPerPoint))
    # accuracyClassifRuptDB.append(np.mean(accuracyRuptureTemp))
    # scoreClassifDB.append(diag.clusterScore(accuracyClassifDB[i],len(np.unique(classClustering)),len(np.unique(testCluster1.labels_)),ratioPenalty,penaltyValue))
    # scoreClassifRuptDB.append(diag.clusterScore(accuracyClassifRuptDB[i],len(np.unique(classClustering)),len(np.unique(testCluster1.labels_)),ratioPenalty,penaltyValue))
    
    
    #DyClee
    cmPerPointDyClee,timePerPointDyClee = doClassifPerPoint(modelClassifDyClee, diagTestScale1, testClassClassif, faultValue,cmPerPointDyClee,timePerPointDyClee, plotDiagPerPoints)
    cmRuptDyClee,timeRuptDyClee = doClassifRupture(modelClassifDyClee, diagTestScale1, testData, testClassClassif, faultValue, rupturePenalty, cmRuptDyClee, timeRuptDyClee, plotDiagPerPoints)
    
    # accuracyClassifDyclee.append(np.mean(accuracyPerPoint))
    # accuracyClassifRuptDyclee.append(np.mean(accuracyRupture))
    # scoreClassifDyclee.append(diag.clusterScore(accuracyClassifDyclee[i],len(np.unique(classClustering)),len(np.unique(DycleeClass)),ratioPenalty,penaltyValue))
    # scoreClassifRuptDyclee.append(diag.clusterScore(accuracyClassifRuptDyclee[i],len(np.unique(classClustering)),len(np.unique(DycleeClass)),ratioPenalty,penaltyValue))

    i=i+1






#Results save
'''
Results1 = accuracyParamKmeans
Results2 = accuracyParamHC
Results3 = accuracyParamDB
Results4 = accuracyParamDB2
Results5 = accuracyParamDyclee

Results6 = scoreParamKmeans
Results7 = scoreParamHC
Results8 = scoreParamDB
Results9 = scoreParamDB2
Results10 = scoreParamDyclee

Results11 = [accuracyKmeans,accuracyHC,accuracyDB,accuracyDyclee]
Results12 = [accuracyClassifKmeans,accuracyClassifHC,accuracyClassifDB,accuracyClassifDyclee]
Results13 = [scoreDB,scoreClassifDB,scoreClassifRuptDB,scoreDyclee,scoreClassifDyclee,scoreClassifRuptDyclee]
Results14 = [accuracyClassifRuptKmeans,accuracyClassifRuptHC,accuracyClassifRuptDB,accuracyClassifRuptDyclee]



if saveResult:
    os.makedirs(resultPath, exist_ok=True)
    np.savetxt(resultPath + '\\accuracyParamKmeans.csv',Results1,delimiter=',')
    np.savetxt(resultPath + '\\accuracyParamHC.csv',Results2,delimiter=',')
    np.savetxt(resultPath + '\\accuracyParamDB.csv',Results3,delimiter=',')
    np.savetxt(resultPath + '\\accuracyParamDB2.csv',Results4,delimiter=',')
    np.savetxt(resultPath + '\\accuracyParamDyclee.csv',Results5,delimiter=',')
    
    np.savetxt(resultPath + '\\scoreParamKmeans.csv',Results6,delimiter=',')
    np.savetxt(resultPath + '\\scoreParamHC.csv',Results7,delimiter=',')
    np.savetxt(resultPath + '\\scoreParamDB.csv',Results8,delimiter=',')
    np.savetxt(resultPath + '\\scoreParamDB2.csv',Results9,delimiter=',')
    np.savetxt(resultPath + '\\scoreParamDyclee.csv',Results10,delimiter=',')
    
    np.savetxt(resultPath + '\\accuracyClusters.csv',Results11,delimiter=',')
    np.savetxt(resultPath + '\\accuracyClassifiCluster.csv',Results12,delimiter=',')
    np.savetxt(resultPath + '\\scoreClassifi.csv',Results13,delimiter=',')
    np.savetxt(resultPath + '\\accuracyClassifiRupt.csv',Results14,delimiter=',')
'''




#Plot accuracy all sets

totalPerPoint = cmPerPointKmeans[0] + cmPerPointKmeans[1] + cmPerPointKmeans[2] + cmPerPointKmeans[3]


cmPercentPerPointKmeans= [100*cmPerPointKmeans[0]/totalPerPoint, 100*cmPerPointKmeans[1]/totalPerPoint, 100*cmPerPointKmeans[2]/totalPerPoint, 100*cmPerPointKmeans[3]/totalPerPoint]
cmPercentPerPointHC= [100*cmPerPointHC[0]/totalPerPoint, 100*cmPerPointHC[1]/totalPerPoint, 100*cmPerPointHC[2]/totalPerPoint, 100*cmPerPointHC[3]/totalPerPoint]
cmPercentPerPointDBSCAN= [100*cmPerPointDBSCAN[0]/totalPerPoint, 100*cmPerPointDBSCAN[1]/totalPerPoint, 100*cmPerPointDBSCAN[2]/totalPerPoint, 100*cmPerPointDBSCAN[3]/totalPerPoint]
cmPercentPerPointDyClee= [100*cmPerPointDyClee[0]/totalPerPoint, 100*cmPerPointDyClee[1]/totalPerPoint, 100*cmPerPointDyClee[2]/totalPerPoint, 100*cmPerPointDyClee[3]/totalPerPoint]

totalPointRupture = cmRuptKmeans[0] + cmRuptKmeans[1] + cmRuptKmeans[2] + cmRuptKmeans[3]
cmPercentRuptKmeans= [100*cmRuptKmeans[0]/totalPointRupture, 100*cmRuptKmeans[1]/totalPointRupture, 100*cmRuptKmeans[2]/totalPointRupture, 100*cmRuptKmeans[3]/totalPointRupture]
totalPointRupture = cmRuptHC[0] + cmRuptHC[1] + cmRuptHC[2] + cmRuptHC[3]
cmPercentRuptHC= [100*cmRuptHC[0]/totalPointRupture, 100*cmRuptHC[1]/totalPointRupture, 100*cmRuptHC[2]/totalPointRupture, 100*cmRuptHC[3]/totalPointRupture]
totalPointRupture = cmRuptDBSCAN[0] + cmRuptDBSCAN[1] + cmRuptDBSCAN[2] + cmRuptDBSCAN[3]
cmPercentRuptDBSCAN= [100*cmRuptDBSCAN[0]/totalPointRupture, 100*cmRuptDBSCAN[1]/totalPointRupture, 100*cmRuptDBSCAN[2]/totalPointRupture, 100*cmRuptDBSCAN[3]/totalPointRupture]
totalPointRupture = cmRuptDyClee[0] + cmRuptDyClee[1] + cmRuptDyClee[2] + cmRuptDyClee[3]
cmPercentRuptDyClee= [100*cmRuptDyClee[0]/totalPointRupture, 100*cmRuptDyClee[1]/totalPointRupture, 100*cmRuptDyClee[2]/totalPointRupture, 100*cmRuptDyClee[3]/totalPointRupture]


meanTimePerPointKmeans = sum(timePerPointKmeans)/len(timePerPointKmeans)
meanTimeRuptKmeans = sum(timeRuptKmeans)/len(timeRuptKmeans)
meanTimePerPointHC = sum(timePerPointHC)/len(timePerPointHC)
meanTimeRuptHC = sum(timeRuptHC)/len(timeRuptHC)
meanTimePerPointDBSCAN = sum(timePerPointDBSCAN)/len(timePerPointDBSCAN)
meanTimeRuptDBSCAN = sum(timeRuptDBSCAN)/len(timeRuptDBSCAN)
meanTimePerPointDyClee = sum(timePerPointDyClee)/len(timePerPointDyClee)
meanTimeRuptDyClee = sum(timeRuptDyClee)/len(timeRuptDyClee)

print("Train set: " + trainDataPath)
print("Test set: " + testDataPath)
print("TP FP FN TN")

print("\nKmeans:")
print("PerPoint")
print(cmPerPointKmeans)
print(cmPercentPerPointKmeans)
print("mean time: " + str(meanTimePerPointKmeans))
print("Rupture:")
print(cmRuptKmeans)
print(cmPercentRuptKmeans)
print("mean time: " + str(meanTimeRuptKmeans))

print("\nHC:")
print("PerPoint")
print(cmPerPointHC)
print(cmPercentPerPointHC)
print("mean time: " + str(meanTimePerPointHC))
print("Rupture:")
print(cmRuptHC)
print(cmPercentRuptHC)
print("mean time: " + str(meanTimeRuptHC))


print("\nDBSCAN:")
print("PerPoint")
print(cmPerPointDBSCAN)
print(cmPercentPerPointDBSCAN)
print("mean time: " + str(meanTimePerPointDBSCAN))
print("Rupture:")
print(cmRuptDBSCAN)
print(cmPercentRuptDBSCAN)
print("mean time: " + str(meanTimeRuptDBSCAN))

print("\nDyClee:")
print("PerPoint")
print(cmPerPointDyClee)
print(cmPercentPerPointDyClee)
print("mean time: " + str(meanTimePerPointDyClee))
print("Rupture:")
print(cmRuptDyClee)
print(cmPercentRuptDyClee)
print("mean time: " + str(meanTimeRuptDyClee))










