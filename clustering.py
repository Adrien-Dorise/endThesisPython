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


def modifyDyclee(g_size):
    file = open('H:\\DIAG_RAD\\Programs\\Dyclee\\2020_11\\bin\\file.txt','r')
    DycleeFile = file.read()
    file.close()
    DycleeFile.find('g_size = ')
    DycleeFile=DycleeFile.replace(DycleeFile[19:24],g_size)
    file = open('H:\\DIAG_RAD\\Programs\\Dyclee\\2020_11\\bin\\file.txt','w')
    file.write(DycleeFile)
    file.close()



# 'H:\\DIAG_RAD\\Programs\\Diagnostic_python\\DiagnosticExample\\ExampleDataSets\\AllDefectAdded.txt'
# 'H:\\DIAG_RAD\\Programs\\Diagnostic_python\\DiagnosticExample\\ExampleDataSets\\All16.txt'

dataChoice = 1 #1 for simulated data: 2 for real datas
trainDataPath = "H:\\DIAG_RAD\\DataSets\\\endThesisValidationData\\simulations\\trainSet"
testDataPath = "H:\\DIAG_RAD\\DataSets\\\endThesisValidationData\\simulations\\testSet\\microLatch"
resultPath = 'H:\\DIAG_RAD\\Results\\IFAC_Safeprocess_2021\\Accuracy\\clusteringAllStats\\test4'
dataIndice = 1
diagDataChoice = 6 # 1 (mean & variance); 2 (mean & frequency); 3 (variance & frequency); 4 (mean & min & max & variance & skewness & kurtosis); 5 (mean & min & max & variance & skewness & kurtosis & freq)
dataName = 'testSet'
dataRange = range(1,1+1)
testRange =  range(1,1+1)
dataParamRange =  range(1,10+1)
penaltyValue = 25
ratioPenalty = 10

saveResult = 0
save=0


dataChoice = 1 #1 for simulated data: 2 for real datas
savePath = 'H:\\DIAG_RAD\\Results\\IFAC_Safeprocess_2021\\multiple_testSets\\3classes\\mean_variance_trainSet'
n_clusters = 3
epsilon = 0.1
plotClustering = 0
plotParallelFeatures = 0




#Init somoe parameters
classificationList = ['Knn', 'naive_bayes', 'decision_tree_classifier', 'random_forest_classifier','svm']
classificationName = ['Knn', 'Naive_Bayes', 'Decision tree', 'Random Forest','SVM']
colors = ['green','red']
className = ['normal','','','','','latch','front de latch up']
if save == 1 or saveResult == 1:
    inp = input('The parameter save equal 1. Do you really wish to save the datas (might overwritte older files)?\nY or N ?\n')
    if (inp == 'N' or inp == 'n' or inp == 'no' or inp == 'No'):
        sys.exit()



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
cmPerPointDyclee,cmRuptDyclee = [0,0,0,0], [0,0,0,0] #TP,FP,FN,TN

i=0
j=0
for indice in dataRange:
    dataIndice = indice
    print('Indice number ' + str(indice))
    timeSerie, features,dataClustering,classClustering, featureChoice, xlabel,ylabel= diag.preprocessing(dataPath = trainDataPath, dataIndice = dataIndice, dataName = dataName,diagDataChoice = diagDataChoice)
    
    
    
    
    clusteringChoice = 'Kmeans'
    testCluster1,testResult1,testPredict1 = diag.clustering(dataClustering,clusteringChoice,epsilon= epsilon,n_clusters=n_clusters,figName='Test_Clust_'+featureChoice,xlabel=xlabel,ylabel=ylabel,save=save,folderPath=savePath,plot=plotClustering,doEvaluation = 0)
    accuracyCluster1,clusterClass1 = diag.doResultClustering(timeSerie,testCluster1.labels_,classClustering,figName = clusteringChoice)
    accuracyKmeans.append(accuracyCluster1)
    #We apply best classification
    accuracyPerPoint=[]
    accuracyRupture=[]
    for testSetNumber in testRange:
        testData, diagTest1 ,diagTestScale1,testClass,featureChoice, xlabel,ylabel= diag.preprocessing(dataPath = testDataPath, dataIndice = testSetNumber,dataChoice = dataChoice, dataName = 'testSet',diagDataChoice = diagDataChoice,save=save)
        featureName = featureChoice
        classifierChoice = 'svm'
        trainClassifi1,trainRoc1,TrainCm1,trainCmAcc1 = diag.classifier(dataClustering,clusterClass1,classifierChoice)
        accuracyPerPointTemp,accuracyRuptureTemp,cmPerPoint,cmRupture = diag.doResultClassification(testData,diagTestScale1,testClass,trainClassifi1)
        accuracyPerPoint.append(accuracyPerPointTemp)
        accuracyRupture.append(accuracyRuptureTemp)
        for j in range(len(cmRupture)):
            cmRuptKmeans[j] += cmRupture[j] 
            cmPerPointKmeans[j] += cmPerPoint[j]
    accuracyClassifKmeans.append(np.mean(accuracyPerPoint))
    accuracyClassifRuptKmeans.append(np.mean(accuracyRuptureTemp))
    
    
    clusteringChoice = 'HC'
    testCluster1,testResult1,testPredict1 = diag.clustering(dataClustering,clusteringChoice,epsilon= epsilon,n_clusters=n_clusters,metric='cosine',figName='Test_Clust_'+featureChoice,xlabel=xlabel,ylabel=ylabel,save=save,folderPath=savePath,plot=plotClustering,doEvaluation = 0)
    accuracyCluster2,clusterClass2 = diag.doResultClustering(timeSerie,testCluster1.labels_,classClustering,figName = clusteringChoice)
    accuracyHC.append(accuracyCluster2)
    accuracyPerPoint=[]
    accuracyRupture=[]
    if len(np.unique(clusterClass2)) <=1:
        clusterClass2[0]=5
    for testSetNumber in testRange:
        testData, diagTest1 ,diagTestScale1,testClass,featureChoice, xlabel,ylabel= diag.preprocessing(dataPath = testDataPath, dataIndice = testSetNumber,dataChoice = dataChoice, dataName = 'testSet',diagDataChoice = diagDataChoice,save=save)
        featureName = featureChoice
        classifierChoice = 'svm'
        trainClassifi1,trainRoc1,TrainCm1,trainCmAcc1 = diag.classifier(dataClustering,clusterClass2,classifierChoice)
        accuracyPerPointTemp,accuracyRuptureTemp,cmPerPoint,cmRupture = diag.doResultClassification(testData,diagTestScale1,testClass,trainClassifi1)
        accuracyPerPoint.append(accuracyPerPointTemp)
        accuracyRupture.append(accuracyRuptureTemp)
        for j in range(len(cmRupture)):
            cmRuptHC[j] += cmRupture[j] 
            cmPerPointHC[j] += cmPerPoint[j]
    accuracyClassifHC.append(np.mean(accuracyPerPoint))
    accuracyClassifRuptHC.append(np.mean(accuracyRuptureTemp))
    
    
    clusteringChoice = 'DBSCAN'
    epsilon = 0.08
    metric='cosine'
    testCluster1,testResult1,testPredict1 = diag.clustering(dataClustering,clusteringChoice,epsilon= epsilon,n_clusters=n_clusters,metric= metric, figName='Test_Clust_'+featureChoice,xlabel=xlabel,ylabel=ylabel,save=save,folderPath=savePath,plot=plotClustering,doEvaluation = 0)
    accuracyCluster3,clusterClass3 = diag.doResultClustering(timeSerie,testCluster1.labels_,classClustering,figName = clusteringChoice)
    accuracyDB.append(accuracyCluster3)
    scoreDB.append(diag.clusterScore(accuracyDB[i],len(np.unique(classClustering)),len(np.unique(testCluster1.labels_)),ratioPenalty,penaltyValue))
    accuracyPerPoint=[]
    accuracyRupture=[]
    if len(np.unique(clusterClass3)) <=1:
        clusterClass3[0]=5
    for testSetNumber in testRange:
        testData, diagTest1 ,diagTestScale1,testClass,featureChoice, xlabel,ylabel= diag.preprocessing(dataPath = testDataPath, dataIndice = testSetNumber,dataChoice = dataChoice, dataName = 'testSet',diagDataChoice = diagDataChoice,save=save)
        featureName = featureChoice
        classifierChoice = 'svm'
        trainClassifi1,trainRoc1,TrainCm1,trainCmAcc1 = diag.classifier(dataClustering,clusterClass3,classifierChoice)
        accuracyPerPointTemp,accuracyRuptureTemp,cmPerPoint,cmRupture = diag.doResultClassification(testData,diagTestScale1,testClass,trainClassifi1)
        accuracyPerPoint.append(accuracyPerPointTemp)
        accuracyRupture.append(accuracyRuptureTemp)
        for j in range(len(cmRupture)):
            cmRuptDBSCAN[j] += cmRupture[j] 
            cmPerPointDBSCAN[j] += cmPerPoint[j]
    
    accuracyClassifDB.append(np.mean(accuracyPerPoint))
    accuracyClassifRuptDB.append(np.mean(accuracyRuptureTemp))
    scoreClassifDB.append(diag.clusterScore(accuracyClassifDB[i],len(np.unique(classClustering)),len(np.unique(testCluster1.labels_)),ratioPenalty,penaltyValue))
    scoreClassifRuptDB.append(diag.clusterScore(accuracyClassifRuptDB[i],len(np.unique(classClustering)),len(np.unique(testCluster1.labels_)),ratioPenalty,penaltyValue))
    
    
    #DyClee
    #Call Dyclee
    print('\nDyClee in progress...')
    modifyDyclee('0.120')
    scaler = preprocessing.MinMaxScaler()
    normalized_X = scaler.fit_transform(dataClustering)
    np.savetxt('H:\\DIAG_RAD\\Programs\\Dyclee\\2020_11\\data\\data.dat',normalized_X,delimiter=',')
    os.system('H:\\DIAG_RAD\\Programs\\Dyclee\\2020_11\\bin\\DyClee H:\\DIAG_RAD\\Programs\\Dyclee\\2020_11\\bin\\file.txt')
    #Use Dyclee class in Python
    DycleeClass = np.genfromtxt('H:\\DIAG_RAD\\Results\\Diagnostic\\DyClee\\DycleeResult.dat',dtype=int)
    DycleeClass = np.subtract(DycleeClass, np.ones(len(DycleeClass),dtype=int))
    while len(DycleeClass) < len(classClustering):
        DycleeClass = np.append(DycleeClass,DycleeClass[len(DycleeClass)-1])
    accuracyCluster4,clusterClass4 = diag.doResultClustering(dataClustering,DycleeClass,classClustering,figName = 'DyClee')
    accuracyDyclee.append(accuracyCluster4)
    scoreDyclee.append(diag.clusterScore(accuracyDyclee[i],len(np.unique(classClustering)),len(np.unique(DycleeClass)),ratioPenalty,penaltyValue))
    print('DyClee done !\n')
    accuracyPerPoint=[]
    accuracyRupture=[]
    if len(np.unique(clusterClass4)) <=1:
        clusterClass4[0]=5
    for testSetNumber in testRange:
        testData, diagTest1 ,diagTestScale1,testClass,featureChoice, xlabel,ylabel= diag.preprocessing(dataPath = testDataPath, dataIndice = testSetNumber,dataChoice = dataChoice, dataName = 'testSet',diagDataChoice = diagDataChoice,save=save)
        featureName = featureChoice
        classifierChoice = 'svm'
        trainClassifi1,trainRoc1,TrainCm1,trainCmAcc1 = diag.classifier(dataClustering,clusterClass4,classifierChoice)
        accuracyPerPointTemp,accuracyRuptureTemp,cmPerPoint,cmRupture = diag.doResultClassification(testData,diagTestScale1,testClass,trainClassifi1)
        accuracyPerPoint.append(accuracyPerPointTemp)
        accuracyRupture.append(accuracyRuptureTemp)
        for j in range(len(cmRupture)):
            cmRuptDyclee[j] += cmRupture[j] 
            cmPerPointDyclee[j] += cmPerPoint[j]
    accuracyClassifDyclee.append(np.mean(accuracyPerPoint))
    accuracyClassifRuptDyclee.append(np.mean(accuracyRupture))
    scoreClassifDyclee.append(diag.clusterScore(accuracyClassifDyclee[i],len(np.unique(classClustering)),len(np.unique(DycleeClass)),ratioPenalty,penaltyValue))
    scoreClassifRuptDyclee.append(diag.clusterScore(accuracyClassifRuptDyclee[i],len(np.unique(classClustering)),len(np.unique(DycleeClass)),ratioPenalty,penaltyValue))

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

accuracyClassifKmeansPlot = np.mean(accuracyClassifRuptKmeans)
accuracyClassifHCPlot = np.mean(accuracyClassifHC)
accuracyClassifDBPlot = np.mean(accuracyClassifDB)
accuracyClassifDycleePlot = np.mean(accuracyClassifDyclee)

accuracyClassifRuptKmeansPlot = np.mean(accuracyClassifRuptKmeans)
accuracyClassifRuptHCPlot = np.mean(accuracyClassifRuptHC)
accuracyClassifRuptDBPlot = np.mean(accuracyClassifRuptDB)
accuracyClassifRuptDycleePlot = np.mean(accuracyClassifRuptDyclee)

clusteringList = ['Kmeans', 'Hierarchical Clustering', 'DBSCAN', 'DyClee']
clustersAccuracy = [accuracyCluster1,accuracyCluster2,accuracyCluster3,accuracyCluster4]
classiAccuracyPerPoint = [accuracyClassifKmeansPlot,accuracyClassifHCPlot,accuracyClassifDBPlot,accuracyClassifDycleePlot]
classiAccuracyRupt = [accuracyClassifRuptKmeansPlot,accuracyClassifRuptHCPlot,accuracyClassifRuptDBPlot,accuracyClassifRuptDycleePlot]
fig1,axs = plt.subplots()
xAxis = range(0,len(clusteringList))
plt.plot(xAxis,clustersAccuracy,label = 'clustering');
plt.plot(xAxis,classiAccuracyPerPoint,label = 'perPoint');
plt.plot(xAxis,classiAccuracyRupt,label = 'Rupture');
plt.title('Clustering algorithms accuracy '+ str(dataName)+ ' ' + str(dataIndice))
plt.xlabel('Clustering algorithm')
plt.ylabel('Accuracy')
plt.ylim(ymax=1)
plt.grid();
plt.legend();
axs.set_xticks(range(0,len(clusteringList)));
axs.set_xticklabels(clusteringList);
diag.saveFigure(save,savePath,'/Classification_accuracy_'+ str(dataIndice)+ '_test_sets.png');
plt.show()











