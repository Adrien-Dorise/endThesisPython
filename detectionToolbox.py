# This program contain multiple diagnostic functions to import in other programs.
# Author: Adrien Dorise
# Date: March 2020

'''
List of Functions :
    watchDog(timeSerie,threshold,classSet=[],save=0,plot=0,xlabel='X',ylabel='Y',figName='',folderPath=''):
    (This function perform a watch dog algorithm on the given time serie)
    
    FourierAnalysis(timeSerie,samplingTime,removeFreqZero=0,plot=0,xlabel='Frequence (Hz)',ylabel='Amplitude',folderPath='',save=0,figName='Signal_Fourier_Transfo'):
    (This function do a frequence analysis of a time serie)
    
    FourierAnalysisPerPoint(timeSerie,samplingTime,removeFreqZero=0,pointAccuracy=50,plot=0,xlabel='Frequence (Hz)',ylabel='Amplitude',folderPath='',save=0,figName='Signal_Fourier_Transfo_PerPoint'):
    (This function do a frequence analysis of a time serie point per point)
    
    timeRupture (data,dataReference=None,penaltyValue=5,save=0,plot=0,xlabel='Time',ylabel='Y',folderPath='')
    (Algorithm finding the ruptures inside a time serie. It uses the Python package ruptures by Charles Truong, Laurent Oudre and Nicolas Vayatis.)


    clustering(data,clusteringChoice='Kmeans',wcss_clusters=10,n_clusters=5,max_iteration=300,n_init=10,random_state=None,save=0,plot=0,xlabel='X',ylabel='Y',folderPath='',scaling=0)
    (Uses a clustering methods to create classes among given data set. It is recommended to first plot the classifier test (elbow method, dendogramm...) to find the optimal number of clusters.
    Implemented methods are 'K-means' and 'Hierarchical clustering'.)


    classifier(data,classes,classifierChoice='Knn',RocDefectValue=5,ensemble_estimators=25,tree_criterion='entropy',knn_n_neighbors=5,random_state=None,save=0,scaling=1,splitSize=0.10,plot=0,xlabel='X',ylabel='Y',classesName='',folderPath='',figName='',randomColors = 0)
    (Uses a classification methods to predict classes among given data set. The data set is splitted into a train set and a test set.
    Implemented methods are 'K-nn', 'SVM', 'Decision tree', 'Random forest', 'Naïve Bayes'.)
    
    def classificationConfusionMatrix(dataSet,dataClass,classifier,abnormalValues,save=0,plot=0,figName='confusion_matrix',folderPath=''):
    (Gives the confusion matrix for the classification of a given dataSet.)
    
    
    fitClassification(point,pointClass,classifier,data=[],scaling=0)
    (Gives the accuracy of a classication algorithm for a given point. If the classification match the real point's class, then it returns 1, else 0.)
    
    
    addPoint(points,data,algo,algoType,pointClass='',clasification_classes_set=None,classification_classesName=None,clustering_n_clusters=None,scaling=0,xlabel='X',ylabel='Y',figName='Diagnostic',save=0, folderPath='')
    (Add new points on an already existing dataset and an exisisting diagnostic. This is only used for plotting.
    It can be used with a classissification algorithm. Then you have to give the lables of the new dataPoint
    It can be used with clustering algoritm. Then you have ot give the number of clusters)

    getClass(indices,classes,classesName)
    (Return the class of a given set of points. The matrix must have been created by the dataGenerator Matlab function
    If there are multiple points, the class chosen is the more present in the set.
    !!!!! Warning: if multiple classes have the same number of iterations, the first one to appear in the classesName array will be chosen!!!!!!!)


    classificationColors(classes_set)
    (Used in the function classifier. This function chooses the colors used for plotting data after a classification algorithm.
    The colors are chosen in order from lowest to highest class value: green, purple,blue,pink,yellow. The value 5 is an exeption as it will always be associatee to the color red (used to differenciate latch class).)


    plotPoints(features_set,scaling=0,xlabel='X',ylabel='Y',figName='Labelled_points',save=0, folderPath='')
    (Plot the points inside the features_set. This is used to better look a data set before doing a classification/clustering algorithm.)


    plotLabelPoints(features_set,classes_set,classesName,scaling=0,colors=['green','purple', 'blue','pink','yellow','orange','red'],xlabel='X',ylabel='Y',figName='Labelled_points',save=0, folderPath='')
    (Plot the points inside the features_set attached with their classes in classes_set. This is used to better look a data set before doing a classification algorithm.)


    plotClassification(features_set,classes_set,classifier,classesName,colors=['green','purple', 'blue','pink','yellow','red','orange'],randomColors=0,withPoints=1,xlabel='X',ylabel='Y',figName='Diagnostic',save=0, folderPath='')
    (This function is used with the function 'classifier'. It plots the points inside the referef class obtained in 'classifier' function. It also shows a colored map to distinguish the different areas of each class.
    Note that the dim of classesName must be the >= of the max value of 'classes_set'. Note2: you can only plot 2 dimensons features.)


    plotClustering(data,clusteringAlgo,n_clusters,xlabel='X',ylabel='Y',figName='Diagnostic',save=0, folderPath='',scaling=0)
    (Plot the labelled points of the give data set. It also shows the area of each clusters along with their centroids of the clustering algorithm trained with the function 'clustering'
    Note: you can only plot 2 dimensons features.)


    importTextData(filePath,numberOfDatas = 2)
    (Import the current data contained in a txt file inside a given folder to a Python data set)


    importMatlabData(folderPath)
    (Import the current data contained in a mat file inside a given folder to a Python data set. This function is designed to work well with the matlab function 'dataGenerator'.)
    
    
    plotCurrent(x,noLatchData,latchData,xlabel='Time (s)',ylabel='Current (mA)',save=0,folderPath='',figName='/graph_consumption_current.png')
    (Plot the current of a given current set. It shows the current with and without the latch up current. Most efficient with the data sets created by the Matlab function 'dataGenerator')
    
    
    plotStatsPerPoint(statsData,x=np.array(0),folderPath='',labels=['Min', 'Max', 'Mean', 'Variance', 'Skewness', 'kurtosis'],figName='/StatsPerpoint_values.png',save=0)
    (Plot the stats of each points of the given data set. The stats showed are: ['Min', 'Max', 'Mean', 'Variance', 'Skewness', 'kurtosis'].)
    
    
    plotROC(roc_fpr, roc_tpr,label=np.array(['No label']),save=0,folderPath='',figName='/ROC_curve.png')
    (Plot the roc curves for classification algorithms. It can plot multiple ROC curves in one figure so it can be usefull to compare multiple algorithms.)
    
    
    plotFeatures(features,featuresName,className,save=0,folderPath='',figName='/Parallel_coordinate.png')
    (Plot the each feature independently of each point in the 'features' parameter. It is used for an quick visualisation of the impact of each feature on the diagnosis.
    You can use it to detect if a feature is interesting to detect a particular class.)
    
    
    stats(data,name=['normal current', 'latch current','other'],save=0,plot=0,folderPath='')
    (Compute stats values of the entire given data. Multiple data sets can be given in parameter, the stats will be given for each one and they will be displayed on the same fig to easily compare them.
    The stats calculated are as follow: ['Min', 'Max', 'Mean', 'Variance', 'Skewness', 'Kurtosis'])
    
    
    statsPerPoint(data,pointAccuracy=20,save=0,plot=0,folderPath='') 
    (Compute the stats of each points contained in the given data. To calculate the stats of each points, we take the points close to them. 
    Note that for firsts lasts value, their stats will remain the same because they all have the same closest points (aka, closest indices in the data set, not closest values.))
    
    
'''
    
from os import path,makedirs
import scipy.io as spio
import numpy as np 
import pandas as pd
import math
from csv import writer

#class MatplotlibMissingError(RuntimeError):
#    pass
#try:
#    import matplotlib.pyplot as plt
#    except ImportError:
#        raise MatplotlibMissingError('This feature requires the optional dependency matpotlib, you can install it using `pip install matplotlib`.')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


#Stats
import scipy.stats as stat


#Data split and scaling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Performance analysis
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve

#Decision/Random tree classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#Knn
from sklearn.neighbors import KNeighborsClassifier

#Naïve Bayes
from sklearn.naive_bayes import GaussianNB

#SVM
from sklearn.svm import SVC
from sklearn.svm import OneClassSVM

#Elliptic envelope
from sklearn.covariance import EllipticEnvelope

#Local Outlier Factor
from sklearn.neighbors import LocalOutlierFactor

#Isolation Forest
from sklearn.ensemble import IsolationForest

#Kmeans
from sklearn.cluster import KMeans

#Hierarchical Clustering
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

#DBSCAN
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

# Time series rupture package
# If package not installed, type in you IDE console 'pip install ruptures --user'
import ruptures as rpt

#Parallel coordinates
from pandas.plotting import parallel_coordinates
 
#Saving/loading models
from joblib import dump,load

#Wavelet
import pywt


class diagDataSet:
    # timeSerie = np.array()
    def __init__(self,dataPath,dataColumnNumber,classPath,classColumn,separator,samplingTime,factor=1):
        self.timeSerie = importTextData(dataPath,dataColumnNumber,separator)
        self.timeSerie = self.timeSerie[:,2]*factor
        self.classData = pd.DataFrame(data=importTextData(classPath,classColumn,separator=',')[:,classColumn],columns=['latch up']).iloc[:,0]
        
        
        



    def watchDog(self):
        return 0
        
        




def waveletAnalysis(timeSerie,wavelet = 'db1',plot=0,xlabel='xLabel',ylabel='yLabel',folderPath='',save=0,figName='Wavelet transformation'):
    '''
    Perform Wavelet transformation

    Parameters
    ----------
    timeSerie : nparray of dim(n,2)
        Data set to be tested The first Column is the time and the second the Y value.
    wavelet : string, optional
        Wavelet type to be used The families availaible are ['haar', 'db', 'sym', 'coif', 'bior', 'rbio', 'dmey', 'gaus', 'mexh', 'morl', 'cgau', 'shan', 'fbsp', 'cmor']. The default is 'db1'.
    plot : int, optional
        Put 1 if you want to plot the figures. The default is 0.
    xlabel : string, optional
        Name for the x axis. The default is 'X'.
    ylabel : string, optional
        Name for the y axis. The default is 'Y'.
    figName : string, optional
        Name of the figure to plot and the file to save. The name of the clustering algo will always be added to it.
        The Default is 'Wavelet transformation' the name of the classifier will always be added to it. Example: 'scaled_datas'
    folderPath : string, optional
        Directory to save the figures in. Note that if the default value is given, figures will not be saved and a warning will show up. The default is ''.
    save : int, optional
        Put 1 if you want to save the figure in the directory given by forlderPath. Put 2 for only the evaluation plot. Put 3 for clusters plot only. Note that you have to plot the figures to save them (plot=1). The default is 0.


    Returns
    -------
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.

    '''
    # print("\nFourierAnalysis in progress ...")
    if np.shape(timeSerie)[1] == 2:
        cA,cD = pywt.dwt(timeSerie[:,1],wavelet=wavelet,mode='sym')
        if plot:
            plt.xlabel('Freq (Hz)')
            plt.ylabel('Amplitude')
            plt.title(figName)
            plt.plot(timeSerie)
            saveFigure(save,folderPath,'/'+figName+'.png')
            plt.show()
        
        # print('FourierAnalysis done!\n')
        return cA,cD
    else:
        print("\nWARNING IN waveletAnalysis: Dimension of timeSerie is incorrect\n")
        return 0,0

    
    
    
    
    

def watchDog(timeSerie,threshold,classSet=[],save=0,plot=0,xlabel='X',ylabel='Y',figName='',folderPath=''):
    '''
    This function perform a watch dog algorithm on the given time serie

    Parameters
    ----------
    timeSerie : nparray of dim(n,2)
        Data set containing the values to be tested by the wathDog algorithm. The first Column is the time and the second the Y value.
    threshold : float
        Value for wich the algorithm will set a flag. Example: 80 for 80mA
    classSet : nparray of dim (n)
        Real class for each value of the data set. It is used to calculate the accuracy of the watchDog algorithm. The normal value is 0. All other value will be treated as a defect. If not given, the functions returns only when the whatcDog sets a flag. The default is []
    plot : int, optional
        Put 1 if you want to plot the figures. The default is 0.
    xlabel : string, optional
        Name for the x axis. The default is 'X'.
    ylabel : string, optional
        Name for the y axis. The default is 'Y'.
    figName : string, optional
        Name of the figure to plot and the file to save. The name of the clustering algo will always be added to it.
        The Default is '' the name of the classifier will always be added to it. Example: 'scaled_datas'
    folderPath : string, optional
        Directory to save the figures in. Note that if the default value is given, figures will not be saved and a warning will show up. The default is ''.
    save : int, optional
        Put 1 if you want to save the figure in the directory given by forlderPath. Put 2 for only the evaluation plot. Put 3 for clusters plot only. Note that you have to plot the figures to save them (plot=1). The default is 0.


    Returns
    -------
    defectDataSet : npArray of dim(m,3)
        Data set containing the value defect values. The columns are [indice,time,amplitude]
    confusionMatrix : nparray of dim(2,2)
        Confusion matrix of the algorithm. Positive is a defect and Negative is a normal value. Lines are predicted classes and columns real classes. cm[0,0] is true positive (good prediction), cm[1,0] is false negative (wrong prediction), cm[0,1] is false positive (wrong prediction), cm[1,1] is true negative (good prediction)
    
    '''
    print("\nWatchDog in progress ...")
    
    
    TP,TN,FP,FN = 0,0,0,0
    defectIndice = np.array(np.where(timeSerie[:,1]>=threshold))
    defectDataSet = np.array([defectIndice[:],timeSerie[defectIndice,0].transpose(),timeSerie[defectIndice,1].transpose()],ndmin=2,dtype=object).transpose()
    
    if len(classSet) != 0:
        if len(classSet) != len(timeSerie):
            print('WARNING IN WATCHDOG: The len of timeSerie and classSet are not equal. Impossible to calculate confusion matrix')
        else:
            for i in range(len(timeSerie)):
                if timeSerie[i,1] >= threshold and classSet[i] > 0:    #True Positive
                    TP = TP + 1;
                elif timeSerie[i,1] <= threshold and classSet[i] == 0:    #True Negative
                    TN = TN + 1;
                elif timeSerie[i,1] >= threshold and classSet[i] == 0:    #False Positive
                    FP = FP + 1;    
                elif timeSerie[i,1] <= threshold and classSet[i] > 0:    #False Negative
                    FN = FN + 1;
                else:
                    print('PROGRAMMING MISTAKE IN WATCHDOG: The data does not fit any condition')
        confusionMatrix=np.array([[TP,FP],[FN,TN]])
    else:
        confusionMatrix=np.array([[0,0],[0,0]])
        
        
    if plot:
        thresholdLine = []
        for i in range(len(timeSerie)):
            thresholdLine.append(threshold)
        plt.plot(timeSerie[:,0],timeSerie[:,1],"b")
        plt.plot(timeSerie[:,0],thresholdLine,"black",label='Threshold',linestyle='--')
        plt.title(figName+' Threhsold')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)        
        plt.legend()
        saveFigure(save,folderPath,figName=('/'+figName+'Threshold.png'))
        plt.show()  
        
        #Plotting accuracy bars
        fig,ax = plt.subplots()
        label = [TP,FN,TN,FP]
        barlist = plt.bar([0,1,2,3],height = [TP,FN,TN,FP],width = 0.8,color = ['green','red','green','red'])
        plt.xticks([0,1,2,3],['True Positive','False Negative','True Negative', 'False Positive'])
        plt.title(figName + ' Confusion Matrix')
        for i in range(0,4): 
            height = barlist[i].get_height()
            ax.text(barlist[i].get_x() + barlist[i].get_width()/2, height, label[i], ha='center', va='bottom') 
        plt.legend([barlist[0],barlist[1]],['Correct prediction','Wrong prediction'])
        saveFigure(save,folderPath,figName = '/'+figName+'_confusionMatrix.png');
        plt.show()

    print("\nWatchDog Done!")
    return defectDataSet,confusionMatrix









def FourierAnalysis(timeSerie,samplingTime,removeFreqZero=0,plot=0,xlabel='Frequence (Hz)',ylabel='Amplitude',folderPath='',save=0,figName='Signal_Fourier_Transfo'):
    '''
    This function do a frequence analysis of a time serie

    Parameters
    ----------
    timeSerie : nparray of dim(n,2)
        Array containing the data set. In must have these 2 sets [Time(s),Data]
    samplingTime : int
        Sampling time used for the data set (in second). Exemple: 0.5s
    removeFreqZero : int, optional
        Put 1 if you want to remove the 0Hz from the resulting frequence. As it can be way higher than other frequence, it is not possible to correctly visualises other values of frequence
    plot : int, optional
        Put 1 if you want to plot the figures. The default is 0.
    xlabel : string, optional
        Name for the x axis. The default is 'Time'.
    ylabel : string, optional
        Name for the y axis. The default is 'Y'.
    folderPath : string, optional
        Directory to save the figures in. Note that if the default value is given, figures will not be saved and a warning will show up. The default is ''.
    save : int, optional
        Put 1 if you want to save the figure in the directory given by forlderPath. Note that you have to plot the figures to save them (plot=1). The default is 0.
    figName : string, optional
        Name of the figure to plot and the file to save. The name of the clustering algo will always be added to it.
        The Default is 'Signal_Fourier_Transfo'. Example: 'scaled_datas'

    Returns
    -------
    freqResult : nparray of dim(i,2)
        Frequence results for the time serie. It gives column 0 is the frequence axis and the column 1 is the amplitude of the corresponding frequence
    object : nparray of dim(i)
        Complex values of the FDT transoformation
    
    

    '''
    # print("\nFourierAnalysis in progress ...")
    if len(np.shape(timeSerie)) == 2:
        if np.shape(timeSerie)[1] == 2:
            fourier = np.fft.fft(timeSerie[:,1])
            if removeFreqZero:
                fourier=np.delete(fourier,0)
            sampleNumber=len(timeSerie[:,1])
            spectre = np.absolute(fourier)*2/sampleNumber
            freq = np.fft.fftfreq(sampleNumber,d=samplingTime)
            if removeFreqZero:
                freq=np.delete(freq,0)
            if plot:
                plt.xlabel('Freq (Hz)')
                plt.ylabel('Amplitude')
                plt.title(figName)
                plt.plot(freq[0:int(len(freq)/2)],spectre[0:int(len(spectre)/2)])
                saveFigure(save,folderPath,'/'+figName+'.png')
                plt.show()
            fourierFreqResult = np.array([freq,spectre]).transpose()
            # print('FourierAnalysis done!\n')
            return fourierFreqResult,fourier
        else:
            print("\nWARNING IN FOURIERANALYSIS: Dimension of timeSerie is incorrect\n")
            return 0,0





def FourierAnalysisPerPoint(timeSerie,samplingTime,removeFreqZero=0,pointAccuracy=50,plot=0,xlabel='Frequence (Hz)',ylabel='Amplitude',folderPath='',save=0,figName='Signal_Fourier_Transfo_PerPoint'):
    '''
    This function do a frequence analysis of a time serie point per point

    Parameters
    ----------
    timeSerie : nparray of dim(n,2)
        Array containing the data set. In must have these 2 sets [Time(s),Data]
    samplingTime : int
        Sampling time used for the data set (in second). Exemple: 0.5s
    pointAccuracy : int, optional
        Accuracy of the stat calculation. This parameter decide how many points are taken from the closest point of our data set (ex if pointAccuracy=20, 10 points will be taken before and after the current point to calculate its stats values). The default is 20.
    removeFreqZero : int, optional
        Put 1 if you want to remove the 0Hz from the resulting frequence. As it can be way higher than other frequence, it is not possible to correctly visualises other values of frequence
    plot : int, optional
        Put 1 if you want to plot the figures. The default is 0.
    xlabel : string, optional
        Name for the x axis. The default is 'Time'.
    ylabel : string, optional
        Name for the y axis. The default is 'Y'.
    folderPath : string, optional
        Directory to save the figures in. Note that if the default value is given, figures will not be saved and a warning will show up. The default is ''.
    save : int, optional
        Put 1 if you want to save the figure in the directory given by forlderPath. Note that you have to plot the figures to save them (plot=1). The default is 0.
    figName : string, optional
        Name of the figure to plot and the file to save. The name of the clustering algo will always be added to it.
        The Default is 'Signal_Fourier_Transfo'. Example: 'scaled_datas'

    Returns
    -------
    spectreMean : nparray of dim(n)
        Array with the mean frequency for each point of the timeSerie
    
    

    '''
    print("\nFourierAnalysisPerPoint in progress ...")
    
    if len(np.shape(timeSerie)) == 2:
       if np.shape(timeSerie)[1] == 2:
        spectreTemp = [];
        # X = np.array(np.zeros((len(data),7)));
        
        fourier = np.fft.fft(timeSerie[:,1])
        if removeFreqZero:
            fourier=np.delete(fourier,0)
        sampleNumber=len(timeSerie[:,1])
        spectre = np.absolute(fourier)*2/sampleNumber
        freq = np.fft.fftfreq(sampleNumber,d=samplingTime)
        if removeFreqZero:
            freq=np.delete(freq,0)
        
        
        for i in range(len(timeSerie)):
            if i < (pointAccuracy)//2:
                indice = i + pointAccuracy
                timeSerieTemp = timeSerie[0:indice]
            elif i+(pointAccuracy//2) > len(timeSerie):
                indice = len(timeSerie) - pointAccuracy + (len(timeSerie)-i)
                timeSerieTemp = timeSerie[indice:len(timeSerie)]
            else:
                timeSerieTemp = timeSerie[i-(pointAccuracy//2):i+(pointAccuracy//2)]
            
            fourierFreqResult,fourier = FourierAnalysis(timeSerieTemp,samplingTime,removeFreqZero)    
            spectreTemp.append(fourierFreqResult[:,1].mean())
    
        if plot:
            plt.xlabel('Points')
            plt.ylabel('Frequence Mean Value')
            plt.title(figName)
            plt.plot(freq,spectre)
            saveFigure(save,folderPath,'/'+figName+'.png')
            plt.show()
        spectreMean = np.array(spectreTemp)    
        print('FourierAnalysisPerPoint done!\n')
        return spectreMean
    else:
            print("\nWARNING IN FOURIERANALYSIS: Dimension of timeSerie is incorrect\n")
            return 0;







def timeRupture (data,dataReference=None,penaltyValue=5,plot=0,xlabel='Indice',ylabel='Y',folderPath='',save=0,figName='time_series_rupture'):
    '''
    Algorithm finding the ruptures inside a time serie. It uses the Python package ruptures by Charles Truong, Laurent Oudre and Nicolas Vayatis.

    Parameters
    ----------
    data : nparray of dim (n,)
        The time serie we want to study
    dataReference : nparray of dim (n,), optional
        If the time serie is a latch up serie, it is possible to plot the the normal current on the same figure using this parameter. The default is None.
    penaltyValue : float, optional
        Penalty value of the alogrithm (>0). The default is 5.
    plot : int, optional
        Put 1 if you want to plot the figures. The default is 0.
    xlabel : string, optional
        Name for the x axis. The default is 'Time'.
    ylabel : string, optional
        Name for the y axis. The default is 'Y'.
    folderPath : string, optional
        Directory to save the figures in. Note that if the default value is given, figures will not be saved and a warning will show up. The default is ''.
    save : int, optional
        Put 1 if you want to save the figure in the directory given by forlderPath. Note that you have to plot the figures to save them (plot=1). The default is 0.
    figName : string, optional
        Name of the figure to plot and the file to save. The name of the clustering algo will always be added to it.
        The Default is 'time_series_rupture'. Example: 'scaled_datas'

    Returns
    -------
    object : ruptures type
        Algo trained by data set.
    breakPoints : nparray of dim (n,)
        Position of ruptures found in data set
    dataBreaks : nparray of object dim (n,2)
        Array containing the portion of data between ruptures. First colunm contains the position of each value in the original data set and second colunm contains values.

    '''
    print('\ntimeRupture in progress ...') 
    dataBreak = []
    dataIndice = []
    ruptureAlgo = rpt.Pelt(model="rbf").fit(data)
    breakPoints = ruptureAlgo.predict(pen=penaltyValue)
    breakPoints = np.insert(breakPoints,0,int(0))
    breakPoints[len(breakPoints)-1] = breakPoints[len(breakPoints)-1] - 1 
    if len(breakPoints) == 1 and breakPoints[0] == len(data):   #If no rupture was found in the data set.
        print('\nWarning in timeRupture: No break point found in the given data set\n')
    else:
        breakPointsTemp = np.insert(breakPoints,0,0)    #We add a 0 to implement all values in dataBreak
        for i in range(len(breakPointsTemp)-1):
            dataBreak.append(data[breakPointsTemp[i]:breakPointsTemp[i+1]])
            dataIndice.append(np.arange(breakPointsTemp[i],breakPointsTemp[i+1],1))
        if plot == 1:
            label = None
            if dataReference is not None:
                plt.plot(dataReference,c='green',label='No Latch Data')
                label = 'Latch Data'
            plt.plot(data,c='blue',label=label)
            for points in breakPoints:
                plt.axvline(x=points - 0.5,
                            color='black',
                            linewidth=1.75,
                            linestyle='--')
            plt.title(figName)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)        
            plt.legend()
            saveFigure(save,folderPath,figName=('/'+figName+'.png'))
            plt.show()  
    print('timeRupture done !\n')
    return ruptureAlgo, np.array(breakPoints), np.array([dataIndice,dataBreak],dtype=object).transpose








def clustering(data,clusteringChoice='Kmeans', metric = 'euclidean',doEvaluation = 1,wcss_clusters=10,n_clusters=5,max_iteration=300,n_init=10,epsilon = 0.3, min_samples = 5,random_state=None,save=0,plot=0,xlabel='X',ylabel='Y',figName='clustering',folderPath=''):
    '''
    Uses a clustering methods to create classes among given data set. It is recommended to first plot the classifier test (elbow method, dendogramm...) to find the optimal number of clusters.
    Implemented methods are 'K-means', 'Hierarchical clustering', 'DBSCAN.

    Parameters
    ----------
    data : nparray of dim (n,m)
        Array containing the features. 'n' dimension is the number of points, 'm' is the number of features. (ex: (1000,2))
    clusteringChoice : string, optional
        Cluster method seleted. Choices are 'Kmeans', 'HC'. The default is 'Kmeans'.
    metric : string, optional
        Metric used for distance absed clustering. You can choose ["euclidean", "l1", "l2", "manhattan", "cosine"] for hierarchical and ["euclidean"", "manhattan, "chebyshe", "minkowski"] for DBSCAN. The default is'euclidean'
    doEvaluation : int, optional
        Put 1 if you want to perform the evaluaiton to help determine the ideal number of cluters. The default is 1
    wcss_clusters : int, optional
        Number of clusters tried for the elbow method when K-means is selected. The default is 10.
    n_clusters : int, optional
        Number of clusters created by the algorithm. Use the classifier test to find optimal number of cluster. The default is 5.
    max_iteration : int, optional
        Maximum number of iterations of the k-means algorithm for a single run. The default is 300.
    n_init : int, optional
        Number of time the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia. The default is 10.
    epsilon : float, optional
        Put the epsilon value for DBSCAN. The maximum distance between two samples for one to be considered as in the neighborhood of the other. This is not a maximum bound on the distances of points within a cluster. This is the most important DBSCAN parameter to choose appropriately for your data set and distance function. The default is 0.3
    min_samples: int, optional
        Parameter for DBSCAN. The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. This includes the point itself. The default is 5
    random_state : int, optional
        Determines random number generation. Use an int to make the randomness deterministic. The default is None.
    plot : int, optional
        Put 1 if you want to plot the figures. The default is 0.
    xlabel : string, optional
        Name for the x axis. The default is 'X'.
    ylabel : string, optional
        Name for the y axis. The default is 'Y'.
    figName : string, optional
        Name of the figure to plot and the file to save. The name of the clustering algo will always be added to it.
        The Default is 'clustering' the name of the classifier will always be added to it. Example: 'scaled_datas'
    folderPath : string, optional
        Directory to save the figures in. Note that if the default value is given, figures will not be saved and a warning will show up. The default is ''.
    save : int, optional
        Put 1 if you want to save the figure in the directory given by forlderPath. Put 2 for only the evaluation plot. Put 3 for clusters plot only. Note that you have to plot the figures to save them (plot=1). The default is 0.


    Returns
    -------
    object : Algo
        Clustering alogrithm trained by the data set.
    evaluationResult : Array
        Array/list containing the results of the evaluation for clustering (ex: Elbow method, Dendogram).
    predictData :  array
        Array containing the prediction of the algorithm
    '''
  
    print('\n'+clusteringChoice+' in progress ...')
    
   
    #Using methods to find optimal number of cluster
    clusterEvaluation = []
    if clusteringChoice == 'Kmeans':
        if doEvaluation:
            #Evaluation using Elbow method
            clusterEvaluation = []
            for i in range(1,wcss_clusters+1):
                clustering = KMeans(n_clusters=i,init='k-means++',max_iter=max_iteration,n_init=n_init,random_state=random_state)
                clustering.fit(data)
                clusterEvaluation.append(clustering.inertia_) 

            plt.plot(range(1,wcss_clusters+1),clusterEvaluation)
            plt.gca().xaxis.set_ticks(range(1,wcss_clusters+1))
            plt.title('The Elbow Method - ' +figName +' '+ clusteringChoice)
            plt.xlabel('Number of Clusters')
            plt.ylabel('WCSS')
            saveFigure(save,folderPath,figName=('/'+figName+'_'+clusteringChoice+'_WCSS.png'))
            plt.show()
        
        clustering = KMeans(n_clusters=n_clusters,init='k-means++',max_iter=max_iteration,n_init=n_init,random_state=random_state)
        
    elif clusteringChoice == 'HC':  
        if doEvaluation:
            #Evaluation using a dendogram
            clusterEvaluation = sch.linkage(data,method='ward')        
            
            sch.dendrogram(clusterEvaluation)
            plt.title('Dendrogram - ' +figName+' '+clusteringChoice)
            plt.xlabel('Datas')
            plt.ylabel('Euclidean distance')
            saveFigure(save,folderPath,figName=('/'+figName+'_'+clusteringChoice+'_dendogram.png'))
            plt.show()
        
        clustering = AgglomerativeClustering(n_clusters=n_clusters,affinity=metric,linkage='average')
    
    elif clusteringChoice == 'DBSCAN':
        if doEvaluation:
            #Evaluation using the distance between closest n points forall data points.
            neighbors = NearestNeighbors(n_neighbors = 2, metric = metric)
            clusterEvaluation = neighbors.fit(data)
            distances, indices = clusterEvaluation.kneighbors(data)
            distances = np.sort(distances, axis=0)
            distances = distances[:,1]
            
            plt.plot(distances)
            plt.title('Distance based method - ' +figName +' '+ clusteringChoice)
            plt.xlabel('Distance (epsilon)')
            plt.ylabel('samples')
            plt.grid()
            saveFigure(save,folderPath,figName=('/'+figName+'_'+clusteringChoice+'_epsilonTest.png'))
            plt.show()
        
        clustering = DBSCAN(eps=epsilon,min_samples=min_samples,metric=metric)
    
    else:
        print('Error in clustering: clusteringChoice unknown')
        
    
    #Applying clustering
    predict_data = clustering.fit_predict(data)
    
    #Plotting
    if plot == 1 or plot == 3:                
        for i in range(0,len(np.unique(predict_data))):
            plt.scatter(data[predict_data == i,0], data[predict_data == i,1],label=('Cluster '+str(i+1)))
        if clusteringChoice == 'Kmeans':    #Plot of centroids
            plt.scatter(clustering.cluster_centers_[:,0],clustering.cluster_centers_[:,1],s=100,c='red',label='Centroids')
        figName = figName + '_' + clusteringChoice
        plt.title(figName)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        saveFigure(save,folderPath,figName=('/'+figName + '_' + str(n_clusters) + '_clusters.png'))
        plt.show()
    print(clusteringChoice+' done !\n')
    return clustering, clusterEvaluation, predict_data





def classifier(data,classes,classifierChoice='Knn',RocDefectValue=1,ensemble_estimators=25,tree_criterion='entropy',knn_n_neighbors=5,random_state=None,save=0,splitSize=0.20,plot=0,xlabel='X',ylabel='Y',classesName='',folderPath='',figName='Classification',randomColors = 0):
    '''
    Uses a classification methods to predict classes among given data set. The data set is splitted into a train set and a test set.
    Implemented methods are 'K-nn', 'SVM', 'Decision tree', 'Random forest', 'Naïve Bayes'.

    Parameters
    ----------
    data : nparray of dim (n,m)
        Array containing the features. 'n' dimension is the number of points, 'm' is the number of features (ex: (1000,2))
    classes : Serie of dim (n,), 'n' dimension must be the same than the parameter data 'n'
        Array with the class value for each point contained in data.
    classifierChoice : string, optional
        Classification method selected. Choices are 'Knn', 'svm', 'decision_tree_classifier, 'random_forest_classifier' and 'naive_bayes'. The default is 'Knn'.
    RocDefectValue : int, optional
        Positive label used to plot the ROC curve. Roc evaluation only works when one label is known as a defect, otherwise,the ROC curve evaluation will be pointless.
    ensemble_estimators : int, optional
        Number of estimators(ex: trees in random forest) used for ensemble algorithms. The default is 25.
    tree_criterion : string, optional
        The function to measure the quality of a split (used in Decision tree/Random forest classifier). Supported criteria are "gini" for the Gini impurity and "entropy" for the information gain. The default is 'entropy'.
    knn_n_neighbors : int, optional
        Number of neighbors used in K-nn algorithm. The default is 5.
    random_state : int, optional
        Determines random number generation. Use an int to make the randomness deterministic. The default is None.
    splitSize : int, optional
        Determines the size of test and train sets. The value given indiquates the percentage of data from the data set used for the test set. The default is 0.25.
    classesName : list of strings, optional
        Names of the different classes used for plotting. Note that the names must follow the same logic than the parameter 'classes' 
        (ex: if a class=4, then the name corresponding must appear at rank 5 of list classesName), otherwise the name will not match the classes leading to false diagnostic!!! 
        Needed if plot=1. The default is ''.
    plot : int, optional
        Put 1 if you want to plot the figures. The default is 0.
    xlabel : string, optional
        Name for the x axis. The default is 'X'.
    ylabel : string, optional
        Name for the y axis. The default is 'Y'.
    folderPath : string, optional
        Directory to save the figures in. Note that if the default value is given, figures will not be saved and a warning will show up. The default is ''.
    save : int, optional
        Put 1 if you want to save the figure in the directory given by forlderPath. Note that you have to plot the figures to save them (plot=1). The default is 0.
    figName : string, optional
        Name of the figure to plot and the file to save. The name of the classifier will always be added to it. The default is'Classification'
 The Default is '' the name of the classifier will always be added to it. Example: 'scaled_datas'
    randomColors : int,optional
        Put 1 if you want matplotlib to choose the colors. Otherwise, the colors will be chosen using the function 'ClassificationColors' to keep the same colors each time. The default is 0.

    Returns
    -------
    object : Algo
        Classification algorithm trained by the data set.
    ROC : list
        Datas of the ROC curve (Can be use to plot multiple ROC curve on one figure).
    confusionMatrix : nparrray of dim (2,2)
        Confusion Matrix of the classifier results.
    accuracy : float
        Accuracy of the classifier regarding the test set.

    '''
    print('\nModel creation of '+classifierChoice+' in progress ...')
    # features contains the features allowing us to diagnostic the datas in a category
    features = pd.DataFrame(data);
    # Splitting dataset into training set and test set
    features_train,features_test,classes_train,classes_test = train_test_split(features,classes, test_size=splitSize,random_state=0);
    features_train = features_train.to_numpy()
    features_test = features_test.to_numpy()
    # Fitting classifier to the training set       
    if classifierChoice == 'Knn':   
        classifier = KNeighborsClassifier(n_neighbors = knn_n_neighbors, metric = 'minkowski', p=2)
    elif classifierChoice == 'svm':
        classifier = SVC(random_state = random_state)
    elif classifierChoice == 'decision_tree_classifier':
        classifier = DecisionTreeClassifier(criterion = tree_criterion, random_state = random_state)
    elif classifierChoice == 'random_forest_classifier':
        classifier = RandomForestClassifier(n_estimators=ensemble_estimators,criterion=tree_criterion,random_state=random_state)
    elif classifierChoice == 'naive_bayes':
        classifier = GaussianNB()    
    else:
        print('Error in classifier: classifierChoice unknown')
    classifier.fit(features_train,classes_train) #Training
    y_pred = classifier.predict(features_test) #Testing
    # Confusion matrix
    cm = confusion_matrix(classes_test,y_pred)
    if len(cm) == 1:
        print('WARNING IN CLASSIFIER: Only one class found in "classes" parameter. Impossible to calculate the confusion matrix accuracy')
        cmAccuracy = 1;
    else:
        cmAccuracy = (cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,1]+cm[0,1]+cm[1,0])
    
    #ROC curve
    roc_fpr, roc_tpr, _ = roc_curve(classes_test, y_pred,pos_label=RocDefectValue);

    if plot == 1:
        if len(classesName) == 0: #If no classes names are give, we cannot continue
            print('WARNING IN CLASSIFIER: No classes names given. Impossible to plot the result')
        else:
            if randomColors != 1:
                color = classificationColors(classes_test)
            else:
                color = ['']    
            figName = figName + '_' + classifierChoice
            plotClassification(features_train, classes_train,classifier ,classesName=classesName,randomColors=randomColors,colors=color,xlabel=xlabel,ylabel=ylabel,figName=(figName+'_(training_set)'),save=save,folderPath=folderPath)
            plotClassification(features_test, classes_test,classifier,classesName=classesName,randomColors=randomColors,colors=color,xlabel=xlabel,ylabel=ylabel,figName=(figName+'_(test_set)'),save=save,folderPath=folderPath)    
            plotROC(roc_fpr,roc_tpr,np.array([classifierChoice]),save=save,folderPath=folderPath,figName=(figName+'_ROC_curve'))
    print(classifierChoice+' model done !\n')
    return classifier,[roc_fpr,roc_tpr],cm, cmAccuracy






def oneClassClassification(data,classifierChoice='OCSVM',random_state=None,withPoints=0,save=0,plot=0,xlabel='X',ylabel='Y',folderPath='',figName='OCC'):
    '''
    Uses a one class classification methods to predict normal classes among given data set. The data set is splitted into a train set and a test set.
    Implemented methods are 'elliptic classification', 'OCSVM', 'LOF', 'isolation forest', 'Auto encoder'.

    Parameters
    ----------
    data : nparray of dim (n,m)
        Array containing the features. 'n' dimension is the number of points, 'm' is the number of features (ex: (1000,2)
    classifierChoice : string, optional
        Classification method selected. Choices are 'Knn', 'svm', 'decision_tree_classifier, 'random_forest_classifier' and 'naive_bayes'. The default is 'Knn'..
    withPoints : int, optional
        Put 1 if you want to plot with the data point on the figure or put 0 if you only want to see the classes borders. The default is 0
    plot : int, optional
        Put 1 if you want to plot the figures. The default is 0.
    xlabel : string, optional
        Name for the x axis. The default is 'X'.
    ylabel : string, optional
        Name for the y axis. The default is 'Y'.
    folderPath : string, optional
        Directory to save the figures in. Note that if the default value is given, figures will not be saved and a warning will show up. The default is ''.
    save : int, optional
        Put 1 if you want to save the figure in the directory given by forlderPath. Note that you have to plot the figures to save them (plot=1). The default is 0.
    figName : string, optional
        Name of the figure to plot and the file to save. The name of the classifier will always be added to it. The default is'Classification'
        The Default is '' the name of the classifier will always be added to it. Example: 'scaled_datas'

    Returns
    -------
    object : Algo
        Classification algorithm trained by the data set.


    '''
    print('\n'+classifierChoice+' in progress ...')
    # features contains the features allowing us to diagnostic the datas in a category
    features_train = np.array(data)
    classes_train = np.zeros(len(features_train))
    # Splitting dataset into training set and test set
    # Fitting classifier to the training set       
    if classifierChoice == 'OCSVM':   
         classifier = OneClassSVM(kernel='rbf',gamma='scale',nu=0.01)
    elif classifierChoice == 'elliptic classification':
        classifier = EllipticEnvelope(support_fraction=0.9, contamination=0.01)
    elif classifierChoice == 'LOF':
        classifier = LocalOutlierFactor(n_neighbors=10,novelty=True,contamination=0.01)
    elif classifierChoice == 'isolation forest':
        classifier = IsolationForest(n_estimators = 50,contamination=0.01)
    else:
        print('Error in classifier: classifierChoice unknown')
    classifier.fit(features_train) #Training

    if plot == 1: 
            figName = figName + '_' + classifierChoice
            plotOCC(features_train, classes_train,classifier,withPoints=withPoints,xlabel=xlabel,ylabel=ylabel,figName=(figName+'_(training_set)'),save=save,folderPath=folderPath)
    print(classifierChoice+' done !\n')
    return classifier




def classificationConfusionMatrix(dataSet,dataClass,classifier,abnormalValues,save=0,plot=0,figName='confusion_matrix',folderPath=''):
    '''
    Gives the confusion matrix for the classification of a given dataSet.

    dataSet
    ----------
    point : array of dim (n,m)
        Features of the dataSet. n is the number of points and m the number of features. m dimension must match the classication algorithm
    dataClass : array of dim(n)
        Real class of each point in the dataSet
    classifier : scikit-learn type
        Classification algorithm already trained
    defectValues : int or array of dm(n)
        List of values considered as abnormal in the dataClass parameter
    plot : int, optional
        Put 1 if you want to plot the figures. The default is 0.
    save : int, optional
        Put 1 if you want to save the figure in the directory given by forlderPath. Put 2 for only the evaluation plot. Put 3 for clusters plot only. Note that you have to plot the figures to save them (plot=1). The default is 0.
    figName : string, optional
        Name of the figure to plot and the file to save. The name of the clustering algo will always be added to it.
        The Default is 'clustering' the name of the classifier will always be added to it. Example: 'scaled_datas'
    folderPath : string, optional
        Directory to save the figures in. Note that if the default value is given, figures will not be saved and a warning will show up. The default is ''.
    
    
    Returns
    -------
    confusionMatrix : nparray of dim(2,2)
        Confusion matrix of the algorithm. Positive is a defect and Negative is a normal value. Lines are predicted classes and columns real classes. cm[0,0] is true positive (good prediction), cm[1,0] is false negative (wrong prediction), cm[0,1] is false positive (wrong prediction), cm[1,1] is true negative (good prediction)
    data : array of dim(n,3)
        array used to verify the alogithm prediction. It is shaped as follow [True Class,Prediction,Result prediction] with the values for Result predicton as 0 for True Pos; 1 for True Neg; 2 for False Pos; 3 for False Negat and 4 for Error

    '''
    verif1,verif2,verif3 = [],[],[]
    verif4=5
    abnormalValues = np.array(abnormalValues)
    TP,TN,FP,FN = 0,0,0,0
    prediction = classifier.predict(dataSet)
    if len(dataClass) != len(dataSet):
        confusionMatrix=np.array([[0,0],[0,0]])
        print('WARNING IN CLASSIFICATION CONFUSION MATRIX: The len of dataSet and dataClass are not equal. Impossible to calculate confusion matrix')
    else:
        for i in range(len(dataSet)):            
            if len(np.where(abnormalValues == prediction[i])[0]) > 0 and len(np.where(abnormalValues == dataClass[i])[0]) > 0:    #True Positive
                TP = TP + 1;
                verif4= 0
            elif len(np.where(abnormalValues == prediction[i])[0]) == 0 and len(np.where(abnormalValues == dataClass[i])[0]) == 0:    #True Negative
                TN = TN + 1;
                verif4= 1
            elif len(np.where(abnormalValues == prediction[i])[0]) > 0 and len(np.where(abnormalValues == dataClass[i])[0]) == 0:    #False Positive
                FP = FP + 1;  
                verif4= 2
            elif len(np.where(abnormalValues == prediction[i])[0]) == 0 and len(np.where(abnormalValues == dataClass[i])[0]) > 0:    #False Negative
                FN = FN + 1;
                verif4= 4
            else:
                print('PROGRAMMING MISTAKE IN CLASSIFICATION CONFUSION MATRIX: The data does not fit any condition')
                verif4= 5
            verif1.append(dataClass[i])
            verif2.append(prediction[i])
            verif3.append(verif4)
        confusionMatrix=np.array([[TP,FP],[FN,TN]])
        if plot:
            #Plotting accuracy bars
            fig,ax = plt.subplots()
            label = [TP,FN,TN,FP]
            barlist = plt.bar([0,1,2,3],height = [TP,FN,TN,FP],width = 0.8,color = ['green','red','green','red'])
            plt.xticks([0,1,2,3],['True Positive','False Negative','True Negative', 'False Positive'])
            plt.title(figName)
            for i in range(0,4): 
                height = barlist[i].get_height()
                ax.text(barlist[i].get_x() + barlist[i].get_width()/2, height, label[i], ha='center', va='bottom') 
            plt.legend([barlist[0],barlist[1]],['Correct prediction','Wrong prediction'])
            saveFigure(save,folderPath,figName = '/'+figName+'.png');
            plt.show()
    return confusionMatrix,np.array([verif1,verif2,verif3]).transpose()





def fitClassification(point,pointClass,classifier):
    '''
    Gives the accuracy of a classication algorithm for a given point. If the classification match the real point's class, then it returns 1, else 0.

    Parameters
    ----------
    point : array of dim (1,2)
        Coordinates of the point to classify.
    pointClass : int
        Real class of the point
    classifier : scikit-learn type
        Classification algorithm already trained

    Returns
    -------
    predictionResult : int
        Result of the prediction. Return 1 if the prediciton is true and 0 if the prediction is wrong

    '''
    if len(np.shape(point)) == 1:    #There are only 1 coordinate but the matrix shape is (2,1) instead of (1,2)
        point = point.reshape(1,-1)
    prediction = classifier.predict(point)
    if prediction == pointClass:
        predictionResult = 1
    else:
        predictionResult = 0
        
    #For Confusion Matrix
    confusionMatrix = [0,0,0,0] #TP,FP,FN,TN
    if prediction == 5 and pointClass == 5:
        confusionMatrix[0] = 1
    if prediction == 5 and pointClass != 5:
        confusionMatrix[1] = 1
    if prediction != 5 and pointClass == 5:
        confusionMatrix[2] = 1
    if prediction != 5 and pointClass != 5:
        confusionMatrix[3] = 1
        
    # print(prediction)
    # print(pointClass)
    # print(confusionMatrix)
        
    return predictionResult,confusionMatrix
    

def confusionMatrixClassifier(point,pointClass,classifier,faultValue, classif = False):
    '''
    Gives the accuracy of a classication algorithm for a given point. If the classification match the real point's class, then it returns 1, else 0.

    Parameters
    ----------
    point : array of dim (1,2)
        Coordinates of the point to classify.
    pointClass : bool
        Real class of the point (true = normal, false = anomaly)
    classifier : scikit-learn type
        Classification algorithm already trained

    Returns
    -------
    Result confusion matrix (TP,FP,FN,TN)
    prediction : int
    '''
    
    confusionMatrix = [0,0,0,0]
    if len(np.shape(point)) == 1:    #There are only 1 coordinate but the matrix shape is (2,1) instead of (1,2)
        point = point.reshape(1,-1)
    prediction = classifier.predict(point)
    
    if(classif):#We do this because depending on the classifier, the prediction value can change
        if(prediction == 1):
            prediction = -1
        if(prediction == 0):
            prediction = 1
        
    
        
    if prediction == 1 and (pointClass != faultValue and pointClass != 100):
        confusionMatrix = np.add(confusionMatrix,[0,0,0,1])
    elif prediction == 1 and pointClass == faultValue:
        confusionMatrix = np.add(confusionMatrix,[0,0,1,0])
    elif prediction == -1 and pointClass != faultValue:
        confusionMatrix = np.add(confusionMatrix,[0,1,0,0])
    elif (prediction == -1 and pointClass == faultValue) or pointClass == 100:
        confusionMatrix = np.add(confusionMatrix,[1,0,0,0])

    # print(str(prediction) + ' / '+ str(pointClass)+' / ')
        
    return confusionMatrix, prediction




def addPoint(points,data,algo,algoType,pointClass='',classification_classes_set=None,classification_classesName=None,clustering_n_clusters=None,colors=['green','purple', 'blue','pink','yellow','red','orange'],randomColors=0,xlabel='X',ylabel='Y',figName='Diagnostic',save=0, folderPath=''):
    '''
    Add new points on an already existing dataset and an exisisting diagnostic. This is only used for plotting.
    It can be used with a classissification algorithm. Then you have to give the lables of the new dataPoint
    It can be used with clustering algoritm. Then you have ot give the number of clusters
    
    Parameters
    ----------
    points : nparray of dim(n,2)
        Coordinates of the new points to add
    data : nparray of dim(n:2)
        DataSets to put on the backround
    algo : scikit learn algorithm
        The algorithm you want to use for the dataSet
    algoType : strinh
        Put 'classifier' or 'clustering' depending of what you want to do'
    pointClass : string, optional
        Classification only : Write the class of the new points. Default is ''. Example:'New lath point'
    classification_classes_set : array of dim(n)
        Classification only : Corresponds to the classes of the data parameter. Default is None
    classification_classesName : list of dim(m)
        Classification only : Names of the different classes. Default is None
    clustering_n_clusters : int
        Clusterig only : Number of clusters from the algorithm. Default is None
    colors : list of strings, optional
        Colors used for plotting the points. The default is ['green','purple', 'blue','pink','yellow','red','orange'].
    randomColors : int,optional
        Put 1 if you want matplotlib to choose the colors. Otherwise, the colors will be chosen using colors parameter. The default is 0.
    plot : int, optional
        Put 1 if you want to plot the figures. The default is 0.
    xlabel : string, optional
        Name for the x axis. The default is 'X'.
    ylabel : string, optional
        Name for the y axis. The default is 'Y'.
    figName : string, optional
        Name of the figure. Used to title the figure and name the file if save=1. The default is 'Diagnostic'. Example: 'scaled_datas'
    save : int, optional
        Put 1 if you want to save the figure in the directory given by forlderPath. Note that you have to plot the figures to save them (plot=1). The default is 0.
    folderPath : string, optional
        Directory to save the figures in. Note that if the default value is given, figures will not be saved and a warning will show up. The default is ''.
    '''
    
    
    if algoType == 'classifier' and (classification_classes_set is None or classification_classesName is None):
        print ('\nWarning in function addPoint: Algo specified is classification, but no classe names given\n')
        return 0
    elif algoType == 'clustering' and clustering_n_clusters is None:
        print ('\nWarning in function addPoint: Algo specified is clustering, but no number of clusters given\n')
        return 0
    
    fig=plt.plot()
    x1, x2 = np.meshgrid(np.arange(start = min(data[:, 0].min(),points[:, 0].min()) - 1, stop = max(data[:, 0].max(),points[:, 0].max()) + 1, step = 0.25),
                         np.arange(start = min(data[:, 1].min(),points[:, 1].min()) - 1, stop = max(data[:, 1].max(),points[:, 1].max()) + 1, step = 0.25))
    
    if algoType == 'classifier':
     
        #Classification algo
#        colors = classificationColors(classification_classes_set)
#        plt.contourf(x1, x2, algo.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
#                      alpha = 0.75, cmap = ListedColormap((colors)))  
       
        classifierValues=algo.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape)
        if randomColors == 1:
            plt.contourf(x1, x2, classifierValues, alpha = 0.75)
        else:
            #We change the scale levels to be sure every class will appear in the plot (otherwise some of them might fuze because too close in value)
            levels = []
            for i in range(int(classifierValues.max()+1)):
                if i in classifierValues:
                    if len(levels) == 0:
                        levels.append(i-0.1)
                    levels.append(i+0.1)
            #It is possible that the classifier does not map all classes if some are under represanted in regards of other classes. In that case, the color map will be incorrect
            #This condition remove non desired colors      
            colorsContour = colors.copy()
            if len(levels) <= len(colors):
                print('\nWARNING IN PlotClassification: The number of classes found by the classifier mismatch the real number of classes. The colors will be placed arbitrary as follow: [NormalCurrent=Green=0, Calculation=Purple=1, I/O=Blue=2, Temperature=Pink=3, Reset=Yellow=4, Latch=Red=5, Front Latch=Orange=6]\n!!!Be careful as this order might not suit your application!!!\n')  
                if ('green' in colorsContour) and (0 not in classifierValues):
                    colorsContour.remove('green')
                if ('purple' in colorsContour) and (1 not in classifierValues):
                    colorsContour.remove('purple')
                if ('blue' in colorsContour) and (2 not in classifierValues):
                    colorsContour.remove('blue')
                if ('pink' in colorsContour) and (3 not in classifierValues):
                    colorsContour.remove('pink')
                if ('yellow' in colorsContour) and (4 not in classifierValues):
                    colorsContour.remove('yellow')
                if ('red' in colorsContour) and (5 not in classifierValues):
                    colorsContour.remove('red')
                if ('orange' in colorsContour) and (6 not in classifierValues):
                    colorsContour.remove('orange')          
            plt.contourf(x1, x2, classifierValues, alpha = 0.75, levels=levels, colors=colorsContour)
        plt.xlim(x1.min(), x1.max())
        plt.ylim(x2.min(), x2.max())
        for i, j in enumerate(np.unique(classification_classes_set)):
            i=int(i)
            j=int(j)
            if len(colors) < i+1:
                colors.insert(i,'black')
            plt.scatter(data[classification_classes_set == j, 0], data[classification_classes_set == j, 1],
                        c = colors[i], label = classification_classesName[j])
        plt.scatter(points[:,0],points[:,1],c='black',s=100,label='New '+pointClass+' point')
        plt.title(figName)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()

    
       
    elif algoType == 'clustering':
        #Clustering algo
        for i in range(0,clustering_n_clusters):
            plt.scatter(data[algo.predict(data) == i,0], data[algo.predict(data) == i,1])
            # plt.scatter(data[classes == 1,0], data[classes == 1,1],s=100,label='Cluster 2')
        
       
        plt.contourf(x1, x2, algo.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),alpha = 0.75) 
        plt.scatter(algo.cluster_centers_[:,0],algo.cluster_centers_[:,1],s=100,c='red',label='Centroids')
        plt.xlim(x1.min(), x1.max())
        plt.ylim(x2.min(), x2.max())
        plt.title(figName)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
    
    else:
        print('\nWARNING IN ADDPOINTS: The \'algoType\' parameter does not correspond to a valide value. Check the documentation to see the possibilities of this parameter\n')
    plt.scatter(points[:,0],points[:,1],c='black',s=75)
    saveFigure(save,folderPath,figName=('/'+figName + '_new_point.png'))
    plt.show()
    return fig





def getClass(indices,classes,classesValuesToFind=[0,1]):
    '''
    Return the class of a given set of points. The matrix must have been created by the dataGenerator Matlab function
    If there are multiple points, the class chosen is the more present in the set.
    !!!!! Warning: if multiple classes have the same number of iterations, the first one to appear in the classesName array will be chosen!!!!!!!
    
    Parameters
    ----------
    indice : nparray of dim(n) or int
        Positons of datas in the data set.
    classes : nparray
        Array containing the class values for the data set.
    classesValuesToFind : array of int
        Values of the possible classes to find

    Returns
    -------
    classValue : int
        Return the class indice in classesName list
    

    '''
    if type(indices)==int:
        return classes[indices]
    
    classCount = np.zeros(len(classesValuesToFind))
    for i in range(len(classesValuesToFind)):   #We count the number of time each class appears in the set
        classCount[i] = np.count_nonzero(classes[indices[0]:indices[1]] == classesValuesToFind[i])
    mainClass = np.array(np.where(classCount == classCount.max())).transpose()
    # if len(mainClass) != 1:
    #     print('Warning in getClass: Multiple classes have same number of iteration. Only one will be selected')
    return mainClass[0][0]







def classificationColors(classes_set):
    '''
    Used in the function classifier. This function chooses the colors used for plotting data after a classification algorithm.
    The colors are chosen in order from lowest to highest class value: green, purple,blue,pink,yellow. The value 5 is an exeption as it will always be associatee to the color red (used to differenciate latch class).

    Parameters
    ----------
    classes_set : npArray of dim (n,)
        Classes value for a given data set.

    Returns
    -------
    colors : list of strings
        The colors chosen for the classes. Usefull with 'plotLabelPoints' function.

    '''
    # Visualising the Training set results
    #Choice of color list
    colorReferences = ['green','red', 'blue','pink','yellow'];
    colors = []
    if type(classes_set) is not np.ndarray:
        classes_set = np.array(classes_set)
    for i in  np.arange(0,max(classes_set)+1):
        i = int(i)
        if i in classes_set and i!=5 and i!=6:
            colors.append(colorReferences[i])
        if i == 5:
            colors.append('red')
        if i == 6:
            colors.append('orange')
    return colors



def plotPoints(features_set,xlabel='X',ylabel='Y',figName='Labelled_points',save=0, folderPath=''):
    '''
    

    Plot the points inside the features_set. This is used to better look a data set before doing a classification/clustering algorithm.

    Parameters
    ----------
    features_set : npArray of dim(n,m)
         Array containing the features. 'n' dimension is the number of points, 'm' is the number of features (ex: (1000,2))
    figName : string, optional
        Name of the figure. Used to title the figure and name the file if save=1. The default is 'Labelled_points'.
    plot : int, optional
        Put 1 if you want to plot the figures. The default is 0.
    xlabel : string, optional
        Name for the x axis. The default is 'X'.
    ylabel : string, optional
        Name for the y axis. The default is 'Y'.
    folderPath : string, optional
        Directory to save the figures in. Note that if the default value is given, figures will not be saved and a warning will show up. The default is ''.
    save : int, optional
        Put 1 if you want to save the figure in the directory given by forlderPath. Note that you have to plot the figures to save them (plot=1). The default is 0.


    Returns
    -------
    fig : fig
        Figure of the plot.

    '''
    fig=plt.figure()
    x1, x2 = np.meshgrid(np.arange(start = features_set[:, 0].min() - 1, stop = features_set[:, 0].max() + 1, step = 0.25),
                          np.arange(start = features_set[:, 1].min() - 1, stop = features_set[:, 1].max() + 1, step = 0.25))
    plt.xlim(x1.min(), x1.max())
    plt.ylim(x2.min(), x2.max())
    plt.scatter(features_set[:,0],features_set[:,1])
    plt.title(figName)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    saveFigure(save,folderPath,figName='/'+figName+'.png')
    plt.show()
    return fig





def plotLabelPoints(features_set,classes_set,classesName,colors=['green','purple', 'blue','pink','yellow','red','orange'],xlabel='X',ylabel='Y',figName='Labelled_points',save=0, folderPath=''):
    '''
    Plot the points inside the features_set attached with their classes in classes_set. This is used to better look a data set before doing a classification algorithm.

    Parameters
    ----------
    features_set : npArray of dim(n,m)
         Array containing the features. 'n' dimension is the number of points, 'm' is the number of features (ex: (1000,2))
    classes_set : erie of dim (n,), 'n' dimension must be the same than the parameter data 'n'
        Array with the class value for each point contained in data.
    classesName : list of strings
        Names of the different classes used for plotting.
        Note that the names must follow the same logic than the parameter 'classes' 
        (ex: if a class=4, then the name corresponding must appear at rank 5 of list classesName), 
        otherwise the name will not match the classes leading to false diagnostic!!!
        Needed if plot=1. The default is ''.
    colors : list of strings, optional
        Colors used for plotting the points. The default is ['green','purple', 'blue','pink','yellow','red'].
    figName : string, optional
        Name of the figure. Used to title the figure and name the file if save=1. The default is 'Labelled_points'.
    plot : int, optional
        Put 1 if you want to plot the figures. The default is 0.
    xlabel : string, optional
        Name for the x axis. The default is 'X'.
    ylabel : string, optional
        Name for the y axis. The default is 'Y'.
    folderPath : string, optional
        Directory to save the figures in. Note that if the default value is given, figures will not be saved and a warning will show up. The default is ''.
    save : int, optional
        Put 1 if you want to save the figure in the directory given by forlderPath. Note that you have to plot the figures to save them (plot=1). The default is 0.


    Returns
    -------
    fig : fig
        Figure of the plot.

    '''
    fig=plt.figure()
    x1, x2 = np.meshgrid(np.arange(start = features_set[:, 0].min() - 1, stop = features_set[:, 0].max() + 1, step = 0.25),
                         np.arange(start = features_set[:, 1].min() - 1, stop = features_set[:, 1].max() + 1, step = 0.25))
    plt.xlim(x1.min(), x1.max())
    plt.ylim(x2.min(), x2.max())
    for i, j in enumerate(np.unique(classes_set)):
        i = int(i)
        j = int(j)
        if len(colors) < i+1:
            colors.insert(i,'black')
        plt.scatter(features_set[classes_set == j, 0], features_set[classes_set == j, 1],
                    c = colors[i], label = classesName[j])
    plt.title(figName)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    saveFigure(save,folderPath,figName='/'+figName+'.png')
    plt.show()
    return fig






def plotClassification(features_set,classes_set,classifier,classesName,colors=['green','purple', 'blue','pink','yellow','red','orange'],randomColors=0,withPoints=1,xlabel='X',ylabel='Y',figName='Diagnostic',save=0, folderPath=''):
    '''
    This function is used with the function 'classifier'. It plots the points inside the referef class obtained in 'classifier' function. It also shows a colored map to distinguish the different areas of each class.
    Note that the dim of classesName must be the >= of the max value of 'classes_set'. Note2: you can only plot 2 dimensons features.
    
    Parameters
    ----------
    features_set : npArray of dim(n,m)
         Array containing the features. 'n' dimension is the number of points, 'm' is the number of features (ex: (1000,2))
    classes_set : erie of dim (n,), 'n' dimension must be the same than the parameter data 'n'
        Array with the class value for each point contained in data.
    classifier : classifier algo
        Classifier algorithm trained using the function 'classifer'.
    classesName : list of strings, optional
        Names of the different classes used for plotting.
        Note that the names must follow the same logic than the parameter 'classes' 
        (ex: if a class=4, then the name corresponding must appear at rank 5 of list classesName), 
        otherwise the name will not match the classes leading to false diagnostic!!!
        Needed if plot=1. The default is ''.
    colors : list of strings, optional
        Colors used for plotting the points. The default is ['green','purple', 'blue','pink','yellow','red','orange'].
    randomColors : int,optional
        Put 1 if you want matplotlib to choose the colors. Otherwise, the colors will be chosen using colors parameter. The default is 0.
    withPoints : int, optional
        Put 1 if you want to plot with the data point on the figure or put 0 if you only want to see the classes borders. The default is 1
    figName : string, optional
        Name of the figure. Used to title the figure and name the file if save=1. The default is 'Diagnostic'.
    scaling : int, optional
        Put 1 if you want to scale the data set. Not scaling can lead to wrong results using clustering algorithm as a feature very high from the others will be dominant. The default is 0.
    plot : int, optional
        Put 1 if you want to plot the figures. The default is 0.
    xlabel : string, optional
        Name for the x axis. The default is 'X'.
    ylabel : string, optional
        Name for the y axis. The default is 'Y'.
    folderPath : string, optional
        Directory to save the figures in. Note that if the default value is given, figures will not be saved and a warning will show up. The default is ''.
    save : int, optional
        Put 1 if you want to save the figure in the directory given by forlderPath. Note that you have to plot the figures to save them (plot=1). The default is 0.


    Returns
    -------
    fig : fig
        Figure of the plot.

    '''
    
    fig=plt.figure()
    x1, x2 = np.meshgrid(np.arange(start = features_set[:, 0].min() - 1, stop = features_set[:, 0].max() + 1, step = 0.25),
                         np.arange(start = features_set[:, 1].min() - 1, stop = features_set[:, 1].max() + 1, step = 0.25))
    
   
    classifierValues=classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape)
    if randomColors == 1:
        plt.contourf(x1, x2, classifierValues, alpha = 0.75)
    else:
        #We change the scale levels to be sure every class will appear in the plot (otherwise some of them might fuze because too close in value)
        levels = []
        for i in range(int(classifierValues.max()+1)):
            if i in classifierValues:
                if len(levels) == 0:
                    levels.append(i-0.1)
                levels.append(i+0.1)
        #It is possible that the classifier does not map all classes if some are under represanted in regards of other classes. In that case, the color map will be incorrect
        #This condition remove non desired colors      
        colorsContour = colors.copy()
        if len(levels) <= len(colors):
            print('\nWARNING IN PlotClassification: The number of classes found by the classifier mismatch the real number of classes. The colors will be placed arbitrary as follow: [NormalCurrent=Green=0, Calculation=Purple=1, I/O=Blue=2, Temperature=Pink=3, Reset=Yellow=4, Latch=Red=5, Front Latch=Orange=6]\n!!!Be careful as this order might not suit your application!!!\n')  
            if ('green' in colorsContour) and (0 not in classifierValues):
                colorsContour.remove('green')
            if ('purple' in colorsContour) and (1 not in classifierValues):
                colorsContour.remove('purple')
            if ('blue' in colorsContour) and (2 not in classifierValues):
                colorsContour.remove('blue')
            if ('pink' in colorsContour) and (3 not in classifierValues):
                colorsContour.remove('pink')
            if ('yellow' in colorsContour) and (4 not in classifierValues):
                colorsContour.remove('yellow')
            if ('red' in colorsContour) and (5 not in classifierValues):
                colorsContour.remove('red')
            if ('orange' in colorsContour) and (6 not in classifierValues):
                colorsContour.remove('orange')          
        plt.contourf(x1, x2, classifierValues, alpha = 0.75, levels=levels, colors=colorsContour)
    plt.xlim(x1.min(), x1.max())
    plt.ylim(x2.min(), x2.max())
    if withPoints == 1:
        for i, j in enumerate(np.unique(classes_set)):
            i=int(i)
            j=int(j)
            if randomColors == 1:
                plt.scatter(features_set[classes_set == j, 0], features_set[classes_set == j, 1],
                        label = classesName[j])
            else:
                if len(colors) < i+1:
                    colors.insert(i,'black')
                plt.scatter(features_set[classes_set == j, 0], features_set[classes_set == j, 1],
                            c = colors[i], label = classesName[j])
    plt.title(figName)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    saveFigure(save,folderPath,figName='/'+figName+'.png')
    plt.show()
    return fig


def plotOCC(features_set,classes_set,classifier,withPoints=1,xlabel='X',ylabel='Y',figName='Diagnostic',save=0, folderPath=''):
    '''
    This function is used with the function 'classifier'. It plots the points inside the referef class obtained in 'classifier' function. It also shows a colored map to distinguish the different areas of each class.
    Note that the dim of classesName must be the >= of the max value of 'classes_set'. Note2: you can only plot 2 dimensons features.
    
    Parameters
    ----------
    features_set : npArray of dim(n,m)
         Array containing the features. 'n' dimension is the number of points, 'm' is the number of features (ex: (1000,2))
    classes_set : serie of dim (n,), 'n' dimension must be the same than the parameter data 'n'
        Array with the class value for each point contained in data. (0 is normal and 1 is anomaly)
    classifier : classifier algo
        Classifier algorithm trained using the function 'classifer'.
    withPoints : int, optional
        Put 1 if you want to plot with the data point on the figure or put 0 if you only want to see the classes borders. The default is 1
    figName : string, optional
        Name of the figure. Used to title the figure and name the file if save=1. The default is 'Diagnostic'.
    scaling : int, optional
        Put 1 if you want to scale the data set. Not scaling can lead to wrong results using clustering algorithm as a feature very high from the others will be dominant. The default is 0.
    plot : int, optional
        Put 1 if you want to plot the figures. The default is 0.
    xlabel : string, optional
        Name for the x axis. The default is 'X'.
    ylabel : string, optional
        Name for the y axis. The default is 'Y'.
    folderPath : string, optional
        Directory to save the figures in. Note that if the default value is given, figures will not be saved and a warning will show up. The default is ''.
    save : int, optional
        Put 1 if you want to save the figure in the directory given by forlderPath. Note that you have to plot the figures to save them (plot=1). The default is 0.


    Returns
    -------
    fig : fig
        Figure of the plot.

    '''
    fig=plt.figure()
    x1, x2 = np.meshgrid(np.arange(start = features_set[:, 0].min() - 1, stop = features_set[:, 0].max() + 1, step = 0.25),
                         np.arange(start = features_set[:, 1].min() - 1, stop = features_set[:, 1].max() + 1, step = 0.25))
    colors = ['green','red', 'orange']
    classesName = ['normal', 'anomaly', 'WARNING']
    classifierValues=classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape)
    #We switch normal value form -1 to 0) 
    for row in classifierValues:
        for i,col in enumerate(row):
            if col == -1:  
                row[i] = 1
            if col == 1:  
                row[i] = 0
    #classifierValues[np.where(classifierValues == -1)[0]] = 1  NOT WORKING
    plt.contourf(x1, x2, classifierValues, alpha = 0.75, levels = [-0.5,0.5,1.5], colors = colors)

    plt.xlim(x1.min(), x1.max())
    plt.ylim(x2.min(), x2.max())
    if withPoints == 1:
        for i, j in enumerate(np.unique(classes_set)):
            i=int(i)
            j=int(j)
            
            plt.scatter(features_set[classes_set == j, 0], features_set[classes_set == j, 1],
                        c = colors[i], label = classesName[j])
    plt.title(figName)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    saveFigure(save,folderPath,figName='/'+figName+'.png')
    plt.show()
    return fig, classifierValues




def plotClustering(data,clusteringAlgo,n_clusters,xlabel='X',ylabel='Y',figName='Diagnostic',save=0, folderPath=''):
    '''
    Plot the labelled points of the give data set. It also shows the area of each clusters along with their centroids of the clustering algorithm trained with the function 'clustering'
    Note: you can only plot 2 dimensons features.
    
    Parameters
    ----------
    data : nparray of dim (n,m)
        Array containing the features. 'n' dimension is the number of points, 'm' is the number of features (ex: (1000,2))
    clusteringAlgo : clustering algorithm
        Algo obtained using the 'clustering' function.
    n_clusters : int
        Number of clusters to show.
    figName : string, optional
        Name of the figure. Used to title the figure and name the file if save=1. The default is 'Diagnostic'.
    xlabel : string, optional
        Name for the x axis. The default is 'X'.
    ylabel : string, optional
        Name for the y axis. The default is 'Y'.
    folderPath : string, optional
        Directory to save the figures in. Note that if the default value is given, figures will not be saved and a warning will show up. The default is ''.
    save : int, optional
        Put 1 if you want to save the figure in the directory given by forlderPath. Note that you have to plot the figures to save them (plot=1). The default is 0.
        

    Returns
    -------
    fig : fig
        Figure of the plot.

    '''
    fig=plt.plot()
  
    for i in range(0,n_clusters):
        plt.scatter(data[clusteringAlgo.predict(data) == i,0], data[clusteringAlgo.predict(data) == i,1])
        # plt.scatter(data[classes == 1,0], data[classes == 1,1],s=100,label='Cluster 2')
    
    x1, x2 = np.meshgrid(np.arange(start = data[:, 0].min() - 1, stop = data[:, 0].max() + 1, step = 0.25),
                         np.arange(start = data[:, 1].min() - 1, stop = data[:, 1].max() + 1, step = 0.25))
    plt.contourf(x1, x2, clusteringAlgo.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),alpha = 0.75) 
    plt.scatter(clusteringAlgo.cluster_centers_[:,0],clusteringAlgo.cluster_centers_[:,1],s=100,c='red',label='Centroids')
    plt.xlim(x1.min(), x1.max())
    plt.ylim(x2.min(), x2.max())
    plt.title(figName)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    saveFigure(save,folderPath,figName=('/'+figName + '_' + str(n_clusters) + '_clusters.png'))
    plt.show()
    return fig




def importTextData(filePath,numberOfDatas = 1,separator = "\t",removeFirstLines = 3):
    '''
    Import the current data contained in a txt file inside a given folder to a Python data set

    Parameters
    ----------
    filePath : string
        Path of the file containing the datas. It must be as follow : [Time(s),Voltage(V),Current(A)] and separed using a tabulation
    numberOfData : int
        Number of different datas in the file. 
    separator : string, optional
        Separator used in the txt file to split the columns. Default is "\t"
    removeFirstLines : int, optional
        Number of lines to remove in the begining of the data file. For example when there is text explaining the data set in the begining. Default is 3
    
    Returns
    -------
    data : nparray of dim (n,3)
        Array containing the datas. It is as follow: [Time(s),Voltage(V),Current(mA)].

    '''
    file = open(filePath,'r');
    contents= file.readlines();
    del contents[0:removeFirstLines]; #Remove non data lines
    data = np.zeros((len(contents),numberOfDatas))
    for i in range(len(contents)):
        dataSplit = contents[i].split(separator);
        for j in range(numberOfDatas):
            data[i,j] = float(dataSplit[j])  
            
    file.close()
    return data;


def importMatlabData(folderPath):
    '''
    (DEPRECIATED: Use importTextData instead)
    Import the current data contained in a mat file inside a given folder to a Python data set. This function is designed to work well with the matlab function 'dataGenerator'.

    Parameters
    ----------
    folderPath : string
        Folder path where are contained the datas.

    Returns
    -------
    data : nparray of dim (n,3)
        Array containing the current sets. The column are organized as follow: [xData, normalCurrent, latchCurrent]
    classData : DataFrame of dim (n,11)
        Array containing the class values of the data set. They are as follow: ['Normal current', 'Calculation', 'I/O consumption', 'Temperature variation', 'reset', 'Latch up', 'Calculation Quantity', 'I/O consumption Quantity', 'Temperature variation Quantity', 'reset Quantity', 'Latch up Quantity'];
    classesName : list of string of dim(11)
        List containing the name of the different classes. They are as follow: ['Normal current', 'Calculation', 'I/O consumption', 'Temperature variation', 'reset', 'Latch up', 'Calculation Quantity', 'I/O consumption Quantity', 'Temperature variation Quantity', 'reset Quantity', 'Latch up Quantity'];


    '''  
    
    
    filePath = folderPath + '\\normalCurrent.mat';
    mat = spio.loadmat(filePath, squeeze_me=True);  
    normalCurrent = mat['normalCurrent']; # array
    
    filePath = folderPath + '/dataSet.mat';
    mat = spio.loadmat(filePath, squeeze_me=True);
    dataSet = mat['dataSet']; # array
    
    filePath = folderPath + '/xData.mat';
    mat = spio.loadmat(filePath, squeeze_me=True);
    xData = mat['xData']; # array
    
    data = np.array([xData, normalCurrent,dataSet]);
    data = data.transpose()
    
    filePath = folderPath + '/statusData.mat';
    mat = spio.loadmat(filePath, squeeze_me=True);
    classData = mat['statusData']; # array
    classesName = ['Normal current', 'Calculation', 'I/O consumption', 'Temperature variation', 'reset', 'Latch up', 'Calculation Quantity', 'I/O consumption Quantity', 'Temperature variation Quantity', 'reset Quantity', 'Latch up Quantity'];
    classData = pd.DataFrame(data=classData,columns=classesName)

    return data, classData, classesName






    
def plotCurrent(x,noLatchData,latchData,xlabel='Time (s)',ylabel='Supply current (mA)',save=0,folderPath='',figName=' '):
    '''
    Plot the current of a given current set. It shows the current with and without the latch up current. Most efficient with the data sets created by the Matlab function 'dataGenerator'

    Parameters
    ----------
    x : nparray of dim (n,)
        Data set for x axis.
    noLatchData : nparray of dim (n,)
        Data set of the normal current.
    latchData : Tnparray of dim (n,)
        Data set of the latch current.
    xlabel : string, optional
        Name for the x axis. The default is 'Time (s)'.
    ylabel : string, optional
        Name for the y axis. The default is 'Current (mA)'.
    folderPath : string, optional
        Directory to save the figures in. Note that if the default value is given, figures will not be saved and a warning will show up. The default is ''.
    save : int, optional
        Put 1 if you want to save the figure in the directory given by forlderPath. Note that you have to plot the figures to save them (plot=1). The default is 0.
    figName : string, optional
        Name of the figure. Used to  name the file if save=1. The default is 'Consumption_current'.


    Returns
    -------
    fig : fig
        Figure of the plot.

    '''
    # Plot of data set
    fig=plt.figure()
    plt.plot(x, latchData, "r", label="Anomaly");
    plt.plot(x, noLatchData, "b", label="Normal");
    plt.title(figName);
    plt.grid(True);
    plt.xlim(min(x)-10,max(x)+10);
    plt.xlabel(xlabel);
    plt.ylim(0, np.max(latchData)+np.max(latchData)*0.2);
    plt.ylabel(ylabel);
    plt.legend();
    saveFigure(save,folderPath,'/'+figName+'.png')
    plt.show();
    return fig







def plotStatsPerPoint(statsData,x=np.array(0),folderPath='',labels=['Min', 'Max', 'Mean', 'Variance', 'Skewness', 'kurtosis'],figName='StatsPerpoint_values',save=0):
    '''
    Plot the stats of each points of the given data set. The stats showed are: ['Min', 'Max', 'Mean', 'Variance', 'Skewness', 'kurtosis'].
    
    Parameters
    ----------
    statsData : npArray of dim(n,m)
        Array containing the stats values of the n points. The column m represents the stat values to plot.
    x : nparray of dim (n,), optional
        Data set for x axis. By default, x takes the n dimension of 'statsData'
    labels : (list of strings), optional
        Names of the stats plotted. The default is ['Min', 'Max', 'Mean', 'Variance', 'Skewness', 'kurtosis'].
    folderPath : string, optional
        Directory to save the figures in. Note that if the default value is given, figures will not be saved and a warning will show up. The default is ''.
    save : int, optional
        Put 1 if you want to save the figure in the directory given by forlderPath. Note that you have to plot the figures to save them (plot=1). The default is 0.
    figName : string, optional
        Name of the figure. Used to title the figure and name the file if save=1. The default is 'StatsPerpoint_values'.

    Returns
    -------
    fig : fig
        Figure of the plot.

    '''
    fig=plt.figure()
    if x.size == 1:
        x=range(statsData.shape[0])
    if len(statsData.shape) == 1:
        statsData = np.reshape(statsData,(statsData.size,1))
    elif len(labels) < statsData.shape[1]:
        print('\n!!!!! WARNING in plotRoc: Not enough labels !!!!!\n')
    for i in range(statsData.shape[1]):
        plt.plot(x,statsData[:,i], label=labels[i])
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title(figName)
        plt.legend(loc='best')
    saveFigure(save,folderPath,'/'+figName+'.png')
    plt.show()
    return fig






def plotROC(roc_fpr, roc_tpr,label=np.array(['No label']),save=0,folderPath='',figName='ROC_curve'):
    '''
    Plot the roc curves for classification algorithms. It can plot multiple ROC curves in one figure so it can be usefull to compare multiple algorithms.

    Parameters
    ----------
    roc_fpr : nparray of dim (n,)
        ROC x axis.
    roc_tpr : nparray of dim (n,)
        ROC y axis.
    label : list of strings, optional
        Names of the classification algorithm attached to the ROC curve. The default is np.array(['No label']).
        folderPath : string, optional
        Directory to save the figures in. Note that if the default value is given, figures will not be saved and a warning will show up. The default is ''.
    save : int, optional
        Put 1 if you want to save the figure in the directory given by forlderPath. Note that you have to plot the figures to save them (plot=1). The default is 0.
    folderPath : string, optional
        Directory to save the figures in. Note that if the default value is given, figures will not be saved and a warning will show up. The default is ''.
    figName : string, optional
        Name of the figure. Used to title the figure and name the file if save=1. The default is 'ROC_curve'.

    Returns
    -------
    fig : fig
        Figure of the plot.

    '''
    
    fig=plt.figure()
    if len(roc_fpr.shape) == 1:
        size = 1   
        roc_fpr = np.array([roc_fpr])
        roc_tpr = np.array([roc_tpr])
    else:
        if label.size < roc_fpr.shape[0]:
            print('\n!!!!! WARNING in plotRoc: Not enough labels !!!!!\n')
        size = roc_fpr.shape[0]
    for i in range(size):
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(roc_fpr[i,:], roc_tpr[i,:], label=label[i])
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve - '+figName)
        plt.legend(loc='best')
    saveFigure(save,folderPath,'/'+figName+'.png')
    plt.show() 
    return fig



def plotFeatures(features,featuresName,className = 'No Class',save=0,folderPath='',figName='Parallel_coordinates'):
    '''
    Plot the each feature independently of each point in the 'features' parameter. It is used for an quick visualisation of the impact of each feature on the diagnosis.
    You can use it to detect if a feature is interesting to detect a particular class.

    Parameters
    ----------
    features : nparray of dim (n,m) n=number of points; m=number of features
        Data set containing the feature points.
    featuresName : string list of dim (m)
        List containing the name of the different features. It must be the same the length that the number of features contained in the 'feature' parameter.
    className : string or list of string of dim(n), optional
        Class name for each points. If no class are given, it is possible to put a string which will be used for all data set. The default is 'No Class'.
    save : int, optional
        Put 1 if you want to save the figure in the directory given by forlderPath. Note that you have to plot the figures to save them (plot=1). The default is 0.
    folderPath : string, optional
        Directory to save the figures in. Note that if the default value is given, figures will not be saved and a warning will show up. The default is ''.
    figName : string, optional
        Name of the figure. Used to title the figure and name the file if save=1. The default is 'Parallel_coordinates'.

    Returns
    -------
    data panda dataFrame of dim (n,3)
        data of the parallel coordinates it contains [data, featuresName, className].
    fig : fig
        Figure of the plot.
    '''
    
    print('\nplotFeatures in progress ...') 
    if len(features[0,:]) != len(featuresName):
        print('Warning: features and featuresName aren\'t the same size!!!')
        print('\nplotFeatures ERROR!\n') 
        return None,None;
    data = pd.DataFrame(data=features,columns=featuresName)
    data['class']=className
    fig=plt.figure()
    axes = parallel_coordinates(data,'class', colormap=plt.get_cmap("Set2"))
    axes.set_title(figName)
    saveFigure(save,folderPath,'/'+figName+'.png')
    plt.show() 
    print('plotFeatures done!\n') 
    return data,fig
    
    






def stats(data,name=['normal current', 'latch current','other'],save=0,plot=0,folderPath='',figName='Stats'):
    '''
    Compute stats values of the entire given data. Multiple data sets can be given in parameter, the stats will be given for each one and they will be displayed on the same fig to easily compare them.
    The stats calculated are as follow: ['Min', 'Max', 'Mean', 'Variance', 'Skewness', 'Kurtosis']

    Parameters
    ----------
    data : nparray of dim (n,m)
        Stats will be computed from this data set. The columns m allow multiple data sets to be computed (ex for 2 data sets of 500 datas each, dim=(500,2)).
    name : list of string, optional
        Names of given data sets. Note that you have to put as many names as the dimension 'm' of 'data' The default is ['normal current', 'latch current','other'].
    save : int, optional
        Put 1 if you want to save the figure in the directory given by forlderPath. Note that you have to plot the figures to save them (plot=1). The default is 0.
    plot : int, optional
        Put 1 if you want to plot the figures. The default is 0.
    folderPath : string, optional
        Directory to save the figures in. Note that if the default value is given, figures will not be saved and a warning will show up. The default is ''.
    figName : string, optional
        Name of the figure. Used to title the figure and name the file if save=1. The default is ''. Example: 'scaled_data'


    Returns
    -------
    statData : nparray of dim (m,7)
        Stats value computed for each data sets. the 7 columns give the following information for the n data sets: ['Min', 'Max', 'Mean', 'Variance', 'Skewness', 'Kurtosis','Standard error of mean', 'Median absolute deviation', 'Geometric standard deviation', 'k-statistic', 'Bayesian confidence interval', 'Wassenrstein distance'].

    '''
    print("\nStats in progress ...")
    
    
    

    # # Previous version of stats. It was possible to do multiple row at once, but didn't have all stats values. 
    # # !!! I NEED TO REWRITE THIS FUNCTION !!!!
    
    # describeStatsTemp = {};
    # describeStats = []
    # # statData
    # labels = ['Min', 'Max', 'Mean', 'Variance', 'Skewness', 'Kurtosis'];
    # figName = [figName+'_Stats_values1',figName+'_Stats_values2']; 
    # width = 0;
    # histXaxis = np.arange(len(labels));  # the label locations
    # rects = {};

    
    # if type(data) is np.ndarray:    #If there is only one matrix of datas
    #     describeStatsTemp[0] = stat.describe(data);
    # else:   #If there are multiple matrix of datas
    #     for i in range(len(data)):
    #         describeStatsTemp[i] = stat.describe(data[i]);
            
    # #We linearize all value contained in describeStatsTemp to have an easier matrix to work with.
    # describeStatsTemp = list(describeStatsTemp.values())
    # describeStatsTemp = np.array(describeStatsTemp)
    # for i in range(len(describeStatsTemp[:,0])):
    #     describeStats.append([describeStatsTemp[i,0],describeStatsTemp[i,1][0],describeStatsTemp[i,1][1],describeStatsTemp[i,2],describeStatsTemp[i,3],describeStatsTemp[i,4],describeStatsTemp[i,5]])
    # describeStats=np.array(describeStats)

    labels=['Min', 'Max', 'Mean', 'Variance', 'Skewness', 'Kurtosis','Standard error of mean', 'Median absolute deviation', 'Geometric standard deviation', 'k-statistic', 'Bayesian confidence interval', 'Wassenrstein distance']
    figName = [figName+'_Stats_values1',figName+'_Stats_values2']; 
    width = 0;
    histXaxis = np.arange(len(labels));  # the label locations
    rects = {};
    
    
    
    
        
    statsDataTemp = list(stat.describe(data))
    statsData=[statsDataTemp[1][0],statsDataTemp[1][1],statsDataTemp[2],statsDataTemp[3],statsDataTemp[4],statsDataTemp[5]]
    statsData.append(stat.sem(data))
    statsData.append(stat.median_abs_deviation(data))
    # c=list(stat.gmean(dataTemp))
    
    statsData.append(stat.gstd(data))
    statsData.append(stat.kstat(data))
    # bayes_mvs
    # e=list(stat.wasserstein_distance(dataTemp,range()))

        



    
    if plot:
        fig,axs = plt.subplots();
        #Plot of first four stats
        width = 1/((1.5**len(statsData)+1))  # the width of the bars
        for i in range(len(statsData)):
            histData = statsData[i]
            for j in range(4):
                histData[j+1] = round(histData[j+1],1);
     
    
            if len(statsData) == 1:
                rects = axs.bar(histXaxis[0:4], histData[1:5], width, label=name[i]);
            else:
                if len(statsData)%2 == 0:
                    rects = axs.bar(histXaxis[0:4] - width/2 + width*i , histData[1:5], width, label=name[i]);
                else:
                    rects = axs.bar(histXaxis[0:4] - width + width*i , histData[1:5], width, label=name[i]);
            for rect in rects:
                height = rect.get_height()
                axs.annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')   
        axs.legend()
        axs.set_title(figName[0])
        axs.set_ylabel('Value');
        axs.set_xticks(histXaxis[0:4]);
        axs.set_xticklabels(labels[0:4]); 
        axs.set_ylim(ymax=statsData[:,1:5].max()+statsData[:,1:5].max()*0.1)
        fig.tight_layout()
        saveFigure(save,folderPath,figName[0])
        plt.show()
        
        
        #Plot of last two stats 
        fig,axs = plt.subplots();
        for i in range(len(statsData)):
            histData = statsData[i]
            histData[5] = round(histData[5],3);
            histData[6] = round(histData[6],3); 
            if len(statsData) == 1:
                rects = axs.bar(histXaxis[0:2], histData[5:7], width, label=name[i]);
            else:
                if len(statsData)%2 == 0:
                    rects = axs.bar(histXaxis[0:2] - width/2 + width*i , histData[5:7], width, label=name[i]);
                else:
                    rects = axs.bar(histXaxis[0:2] - width + width*i , histData[5:7], width, label=name[i]);
            for rect in rects:
                height = rect.get_height()
                axs.annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')  
        axs.legend()
        axs.set_title(figName[1])
        axs.set_ylabel('Current (mA)');
        axs.set_xticks(histXaxis[0:2]);
        axs.set_xticklabels(labels[4:6]); 
        axs.set_ylim(statsData[:,5:7].min()+statsData[:,5:7].min()*0.1,ymax=statsData[:,5:7].max()+statsData[:,5:7].max()*0.3)
        fig.tight_layout()
        saveFigure(save,folderPath,figName[1])
        plt.show()
        
        
    
    print('Stats done!\n')
    return statsData     
 






def statsPerPoint(data,pointAccuracy=20,save=0,plot=0,folderPath='',figName='StatsPerPoint'):
    '''
    Compute the stats of each points contained in the given data. To calculate the stats of each points, we take the points close to them. 
    Note that for firsts lasts value, their stats will remain the same because they all have the same closest points (aka, closest indices in the data set, not closest values.)
    UPDATE: Now instead of taking the window before and after, we are taking the points after the "i" indice. The last points of the data set are not taken into consideration
    
    Parameters
    ----------
    data : nparray of dim (n,)
        Stats will be computed from this data set of n points. It is better to have a time serie.
    pointAccuracy : int, optional
        Accuracy of the stat calculation. This parameter decide how many points are taken from the closest point of our data set (ex if pointAccuracy=20, 10 points will be taken before and after the current point to calculate its stats values). The default is 20.
    save : int, optional
        Put 1 if you want to save the figure in the directory given by forlderPath. Note that you have to plot the figures to save them (plot=1). The default is 0.
    plot : int, optional
        Put 1 if you want to plot the figures. The default is 0.
    folderPath : string, optional
        Directory to save the figures in. Note that if the default value is given, figures will not be saved and a warning will show up. The default is ''.
    figName : string, optional
        Name of the figure. Used to title the figure and name the file if save=1. The default is ''. Example: 'scaled_data'


    Returns
    -------
    statData : nparray of dim (n,11)
        ['index','min', 'max', 'mean', 'variance', 'skewness', 'kurtosis','standard error of mean', 'median absolute deviation', 'geometric standard deviation', 'k statistic'])      

    '''
    print("\nStatsPerPoint in progress ...")
    statsDataTemp = {};
    a={}
    b={}
    c={}
    d={}
    e={}
    statsData = np.array(np.zeros((len(data),11)));
    labels=['Min', 'Max', 'Mean', 'Variance', 'Skewness', 'Kurtosis','Standard error of mean', 'Median absolute deviation', 'Geometric standard deviation', 'k-statistic', 'Bayesian confidence interval', 'Wassenrstein distance']
    for i in range(len(statsData)):
        # if i < (pointAccuracy)//2:
        #     indice = i + pointAccuracy
        #     dataTemp = data[0:indice]
        # elif i+(pointAccuracy//2) > len(data):
        #     indice = len(data) - pointAccuracy + (len(data)-i)
        #     dataTemp = data[indice:len(data)]
        # else:
        #     dataTemp = data[i-(pointAccuracy//2):i+(pointAccuracy//2)]
        if i + pointAccuracy < len(data):
            dataTemp = data[i:i+pointAccuracy]
        
        statsDataTemp = list(stat.describe(dataTemp))
        a=stat.sem(dataTemp)
        b=stat.median_absolute_deviation(dataTemp)
        # c=list(stat.gmean(dataTemp))
        # c=stat.gstd(dataTemp)
        c=0
        d=stat.kstat(dataTemp)
        # bayes_mvs
        # e=list(stat.wasserstein_distance(dataTemp,range()))

        
        statsData[i]=[i,statsDataTemp[1][0],statsDataTemp[1][1],statsDataTemp[2],statsDataTemp[3],statsDataTemp[4],statsDataTemp[5],a,b,c,d]

    if plot:
        plotStatsPerPoint(statsData[:,1:4],statsData[:,0],labels=labels[0:3],save=save,folderPath=folderPath,figName=figName+'_values1')
        plotStatsPerPoint(statsData[:,4],statsData[:,0],labels=np.reshape(np.array(labels[3]),1),save=save,folderPath=folderPath,figName=figName+'_values2')
        plotStatsPerPoint(statsData[:,5:7],statsData[:,0],labels=labels[4:6],save=save,folderPath=folderPath,figName=figName+'_values3')
        
        
        plotStatsPerPoint(statsData[:,7],statsData[:,0],labels=np.reshape(np.array(labels[6]),1),save=save,folderPath=folderPath,figName=figName+'_values4')
        plotStatsPerPoint(statsData[:,8],statsData[:,0],labels=np.reshape(np.array(labels[7]),1),save=save,folderPath=folderPath,figName=figName+'_values5')
        plotStatsPerPoint(statsData[:,9],statsData[:,0],labels=np.reshape(np.array(labels[8]),1),save=save,folderPath=folderPath,figName=figName+'_values6')
        plotStatsPerPoint(statsData[:,10],statsData[:,0],labels=np.reshape(np.array(labels[9]),1),save=save,folderPath=folderPath,figName=figName+'_values7')

    
    print('StatsPerPoint done!\n')
    return statsData







def saveFigure(save=0,folderPath='',figName='/figure.png'):
    '''
    Used to save the current figure inside a given folder

    Parameters
    ----------
    save : int, optional
        Put 1 if you want to save the figure in the directory given by forlderPath. Note that you have to plot the figures to save them (plot=1). The default is 0.
    folderPath : string, optional
        Directory to save the figures in. Note that if the default value is given, figures will not be saved and a warning will show up. The default is ''.
    figName : string, optional
        Name of the figure. Used to title the figure and name the file if save=1. The default is '/figure.png'.


    Returns
    -------
    None.

    '''
    if save:
        if folderPath == '':
            print('Problem for' + figName +': No folder path given for saving the plot')
        else:
            if not path.exists(folderPath):
                makedirs(folderPath)
            figName = folderPath + figName;
            plt.savefig(figName,dpi=175,bbox_inches = 'tight');
            print('file save in: ' + figName)


def saveCSV(path,fileName, data):
    '''
    

    Parameters
    ----------
    path : TYPE
        Folder path.
    fileName : TYPE
        exemple: resultOfTest.
    data : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    makedirs(path, exist_ok=True)
    with open(path + '/' + fileName + '.csv', 'w') as f: 
        write = writer(f) 
        write.writerow(data)
    
    
    

def saveModel(model,path):
    # serialize model to joblib
    dump(model,path+".joblib")
    print("Saved model to disk")
    
    
    
def loadModel(path):    
    #Load a model
    return load(path+".joblib")





def createDataFrame(indexName):
    dictionnary = {'dataPath' : 'a',
                   'dataFolder' : '',
                   'timeSerie' : [0],
                   'diagData' : [0],
                   'scaler' : StandardScaler(),
                   'scaledDiagData' : [0],
                   'class' : [0],
                   'allFeatures' : 0 ,
                   'featureName' : '',
                   'watchDog' : 0,
                   'timeRupture' : 0,
                   'classification' : 0,
                   'clustering' : 0,
                   'results' : 0
                   }
    
    df=pd.DataFrame(dictionnary,index=indexName)
    # df.at[indexName,'classification']={'algorithmChoice' : [0],'object' : [0],'truc' : [0],'machin' : [0]}
    return df
    
  



def plotPredictionOnTimeSerie(timeSerie,prediction,acquiTime = 1,figName = 'Clustering'):
    xData =timeSerie[:,0]
    plotData=timeSerie[:,1]
    plt.figure()
    for i in range(np.min(np.unique(prediction)),np.max(np.unique(prediction))+1):
        plt.scatter(xData[prediction == i], plotData[prediction == i],label=('Prediction '+str(i)))
    plt.ylim(bottom = 0,top = np.max(plotData)+10)
    plt.legend()
    plt.title(str(figName) + ' predictions on time series')
    plt.show()








def preprocessing(dataPath,dataIndice = 1,dataChoice = 1,diagDataChoice = 2, windowSize = 20,dataName = 'dataSet',dataColumnChoice=2, plotFeatures = 0,savePath='',save=0,addNewLatchClass = 0):
    colors = ['green','red']
    plotStats = 0
    plotFrequency = 0
    className = ['normal','latch','','','','latch','front de latch up']
    
    if dataChoice == 2:
      #Folder path initialisation
    # dataPath = 'H:\\DIAG_RAD\\Programs\\Diagnostic_python\\DiagnosticExample\\ExampleDataSets'   
    # dataPath = dataPath + '\\AllDefectAdded.txt'
        separator = '\t'
        samplingDataTime = 75
        # testDataPath = dataPath + '\\All16.txt'
    

    
    #Data path initialisation
    if dataChoice == 1:
        dataPath = dataPath + '/datas'+ str(dataIndice) +'/diagData.txt'
        dataFolder = dataPath.split('/diagData.txt')[0];
        separator = ','
        samplingDataTime = 1000

    
    
    
    
    #Import Data
    timeSerie = importTextData(dataPath,3,separator)
    if separator == '\t': #Data coming from dataAcquisition and need to bu put in mA
        timeSerie[:,2] = timeSerie[:,2]*1000 
        plotCurrent(timeSerie[:,0],timeSerie[:,2],timeSerie[:,2],save=save,folderPath=savePath,figName = dataName + ' ')
    elif separator == ',':
        plotCurrent(timeSerie[:,0],timeSerie[:,1],timeSerie[:,2],save=save,folderPath=savePath,figName = dataName + ' ')
    

    
    
    #Class creation for trainSet
    if separator == '\t': #Data coming from dataAcquisition 
        dataClass = []
        for i in range(0,9989):
            dataClass.append(0)
        for i in range(9989,len(timeSerie)):
            dataClass.append(5)    
        dataClass = pd.DataFrame(data=dataClass,columns=['Latch up']).iloc[:,0]
    else:
        dataClass = pd.DataFrame(data=importTextData(dataFolder+'/statusData.txt',6,separator=',')[:,5],columns=['latch up']).iloc[:,0]
   
    
        
    #Adding new latch class
    if addNewLatchClass:
        colors.append('orange')
        firstLatchIndex=np.where(dataClass==5)[0][0]
        dataClass[firstLatchIndex:firstLatchIndex+45]=6

        
    
    
   
    #Finding Features
        #Stats
    statDataPerPoint = statsPerPoint(timeSerie[:,dataColumnChoice], pointAccuracy = windowSize, save=save,folderPath=savePath,plot=plotStats,figName=dataName + '_set_StatsPerPoint')
    [dataMinPerPoint,dataMaxPerPoint,dataMeanPerPoint,dataVariancePerPoint,dataSkewnessPerPoint,dataKurtosisPerPoint,dataSEMPerPoint,dataMedianPerPoint,dataStdPerPoint,dataKstatPerPoint] = [statDataPerPoint[:,1],statDataPerPoint[:,2],statDataPerPoint[:,3],statDataPerPoint[:,4],statDataPerPoint[:,5],statDataPerPoint[:,6],statDataPerPoint[:,7],statDataPerPoint[:,8],statDataPerPoint[:,9],statDataPerPoint[:,10]]
        
        #Frequency
    dataFourier = FourierAnalysisPerPoint(timeSerie[:,[0,dataColumnChoice]], samplingDataTime,removeFreqZero=1, pointAccuracy = windowSize, plot=plotFrequency,save=save,folderPath=savePath,figName=dataName + '_frequency')
    
    
    #DiagData Creation
    if diagDataChoice == 1:
        diagData =  np.array([dataVariancePerPoint,dataMeanPerPoint]).transpose()
        ylabel = 'Mean current (mA)'
        xlabel = 'Variance'
        featureChoice = 'variance'
        featureName = 'Variance'
    elif diagDataChoice == 2:
        diagData =  np.array([dataFourier,dataMeanPerPoint]).transpose()
        ylabel = 'Mean current (mA)'
        xlabel = 'Fourier transformation'
        featureChoice = 'fourier'
        featureName = 'Fourier'
    elif diagDataChoice == 3:
        diagData =  np.array([dataVariancePerPoint,dataFourier]).transpose()
        ylabel = 'Fourier transformation'
        xlabel = 'Variance'
        featureChoice = 'var_fourier'
        featureName = 'Variance / Fourier'
    elif diagDataChoice == 4:
        diagData =  np.array([dataMinPerPoint,dataMaxPerPoint,dataMeanPerPoint,dataVariancePerPoint,dataSkewnessPerPoint,dataKurtosisPerPoint]).transpose()
        ylabel = 'Y'
        xlabel = 'X'
        featureChoice = 'stats'
        featureName = 'Stats'
    elif diagDataChoice == 5:
        diagData =  np.array([dataMinPerPoint,dataMaxPerPoint,dataMeanPerPoint,dataVariancePerPoint,dataSkewnessPerPoint,dataKurtosisPerPoint,dataFourier]).transpose()
        ylabel = 'Y'
        xlabel = 'X'
        featureChoice = 'stats_fourier'
        featureName = 'Stats / Fourier'
    elif diagDataChoice == 6:
            diagData =  np.array([dataMinPerPoint,dataMaxPerPoint,dataMeanPerPoint,dataVariancePerPoint,dataSEMPerPoint,dataMedianPerPoint,dataKstatPerPoint]).transpose()
            ylabel = 'Y'
            xlabel = 'X'
            featureChoice = 'bigStats'
            featureName = 'bigStats'
    
        
    
    
    
    
    #Scaling of datas
    sc1 = StandardScaler();
    diagDataScale = sc1.fit_transform(diagData);  
    
    
    
    
    #Plotting of features
    if plotFeatures:
        #Non scaled datas
            plotLabelPoints(diagData, dataClass, className,figName=dataName + '_NonScaled_'+featureChoice,colors=colors,xlabel=xlabel,ylabel=ylabel,save=save,folderPath=savePath)
        #Scaled datas
            plotLabelPoints(diagDataScale, dataClass, className,figName=dataName + '_Scaled_'+featureChoice,colors=colors,xlabel=xlabel,ylabel=ylabel,save=save,folderPath=savePath)
            
    return timeSerie, diagData, diagDataScale, dataClass,featureChoice,xlabel,ylabel




def statExtraction(timeSerieScale,windowSize,diagDataChoice):
    #Finding Features
        #Stats
    statDataPerPoint = statsPerPoint(timeSerieScale, pointAccuracy = windowSize)
    [dataMinPerPoint,dataMaxPerPoint,dataMeanPerPoint,dataVariancePerPoint,dataSkewnessPerPoint,dataKurtosisPerPoint,dataSEMPerPoint,dataMedianPerPoint,dataStdPerPoint,dataKstatPerPoint] = [statDataPerPoint[:,1],statDataPerPoint[:,2],statDataPerPoint[:,3],statDataPerPoint[:,4],statDataPerPoint[:,5],statDataPerPoint[:,6],statDataPerPoint[:,7],statDataPerPoint[:,8],statDataPerPoint[:,9],statDataPerPoint[:,10]]
        
    
    #DiagData Creation
    if diagDataChoice == 1:
        diagData =  np.array([dataVariancePerPoint,dataMeanPerPoint]).transpose()
        ylabel = 'Mean current (mA)'
        xlabel = 'Variance'
        featureChoice = 'variance'
        featureName = 'Variance'
    elif diagDataChoice == 2:
        print("NOT A VALID DATA CHOICE")
    elif diagDataChoice == 3:
        print("NOT A VALID DATA CHOICE")
    elif diagDataChoice == 4:
        diagData =  np.array([dataMinPerPoint,dataMaxPerPoint,dataMeanPerPoint,dataVariancePerPoint,dataSkewnessPerPoint,dataKurtosisPerPoint]).transpose()
        ylabel = 'Y'
        xlabel = 'X'
        featureChoice = 'stats'
        featureName = 'Stats'
    elif diagDataChoice == 5:
        print("NOT A VALID DATA CHOICE")
    elif diagDataChoice == 6:
            diagData =  np.array([dataMinPerPoint,dataMaxPerPoint,dataMeanPerPoint,dataVariancePerPoint,dataSEMPerPoint,dataMedianPerPoint,dataKstatPerPoint]).transpose()
            ylabel = 'Y'
            xlabel = 'X'
            featureChoice = 'bigStats'
            featureName = 'bigStats'

    
    return diagData
    
    
    

def doResultClustering(timeSerie,clusterPredict,realClass,plot=0,xlabel='X',ylabel='Y',folderPath='',save=0,figName='resultClustering'):
    
    #Plotting on timeSerie
    if plot:
        plotPredictionOnTimeSerie(timeSerie,clusterPredict,figName = figName)
    
    
    
    #Expert opinion !
    #We want to know predominent prediction for each class
    prediction = clusterPredict
    countClass = np.zeros((len(np.unique(realClass)),len(np.unique(prediction))+1),dtype=int)
    countClass[:,0] = np.unique(realClass)
    i,j = 0,0
    for iClass in np.unique(realClass):
        
        for iPred in np.unique(prediction):
            j=j+1
            countClass[i,j] = np.sum(prediction[realClass == iClass] == iPred)
        i=i+1
        j=0
    
    #Plotting of occurence of each predicito for each class
    #CANNOT PLOT MORE THAN 2 CLASSES, REWRITE TO AVOID THIS PROBLEM 
    if plot:
        i=0
        width = 0.35
        fig,ax=plt.subplots()
        
        ax.bar(np.arange(len(countClass[0,1:])) - width/2, countClass[0,1:], width, label='Normal data')
        ax.bar(np.arange(len(countClass[1,1:])) + width/2, countClass[1,1:], width, label='latch-up')
        
        ax.set_xlabel('Prediction classes')
        ax.set_ylabel('Occurence')
        ax.set_title(str(figName) + ' predictions')
        ax.set_xticks(np.arange(len(countClass[0,1:])))
        xBarLabel = []
        for i in range(len(np.unique(prediction))):
            xBarLabel.append(str(np.unique(prediction)[i]))
        ax.set_xticklabels(xBarLabel)
        ax.legend()
        plt.show()
    
    
    
    
    #Now we can give a proper opinion
    # pred automatic
    predictionAdjusted = np.zeros(len(prediction))
    for i in range(len(prediction)):
        for j in range(len(countClass[0,1:])):
            if prediction[i] == j:
                temp = np.where(countClass == max(countClass[:,j+1]))[0][0]
                predictionAdjusted[i] = countClass[temp,0]
    
    
    
    #pred à la mano
    # predictionAdjusted = np.zeros(len(prediction))
    # for i in range(len(prediction)):
        
    #     if prediction[i] == 0:
    #         predictionAdjusted[i] = 5
    #     if prediction[i] == 1:
    #         predictionAdjusted[i] = 0
    #     if prediction[i] == 2:
    #         predictionAdjusted[i] = 0
    #     if prediction[i] == 3:
    #         predictionAdjusted[i] = 5
    
    
    
    #Accuracy calculation
    goodPrediction = 0
    wrongPrediction = 0
    for i in range(len(predictionAdjusted)):
        if predictionAdjusted[i] == realClass[i]:
            goodPrediction = goodPrediction +1
        else:
            wrongPrediction = wrongPrediction +1
    accuracy = goodPrediction / (goodPrediction+wrongPrediction)
    print('Clustering accuracy is: ' + str(accuracy))
    if plot:
        plt.bar([0,1],[goodPrediction,wrongPrediction])
        plt.title(str(figName) + ' accuracy')
        
        
        plt.show()
    return accuracy,predictionAdjusted





def doResultClassification(testSerie,diagTestScale,realTestClass,classifier,plot=0,xlabel='X',ylabel='Y',folderPath='',save=0,figName='resultClassification',savePath='',classifierChoice='classification',featureChoice=''):
    if plot == 0:
        plotTimeRupt = 0
        plotDiagRupture = 0
        plotDiagPerPoints = 0
    elif plot == 1 :
        plotTimeRupt = 1
        plotDiagRupture = 1
        plotDiagPerPoints = 1
    className = ['normal','Calculation','I/O consumption','Temperature variation','reset','latch','front de latch up']
    
    classifierAccuracy1 = []
    correctPrediction1 = 0
    wrongPrediction1 = 0
    cmRupture = [0,0,0,0]

            
    #Time Series Rupture
    timeRuptureAlgo,timeRuptureBreakPoints,timeRuptureDataBreak = timeRupture(testSerie[:,2],penaltyValue=5,folderPath=savePath,ylabel='Current',xlabel='Time (s)',save=save,plot=plotTimeRupt)

    
    #Test using rupture package        
    print('\nAccuracy Calculation in progress...\n')
    for i in range(len(timeRuptureBreakPoints)-1):
        figName = classifierChoice + '_test_' + featureChoice + '_rupture_confusion_matrix_' + str(i+1) 
        
        indices1 = [timeRuptureBreakPoints[i],timeRuptureBreakPoints[i+1]]   
        classValue1, classValueString1 = getClass(indices1,realTestClass,className)   #getting the real class for the set of points
        points = np.array(diagTestScale[indices1,0].mean(),ndmin=2)
        for featureColumn in range(1,diagTestScale.shape[1]):
            points = np.append(points,np.array(diagTestScale[indices1,0].mean(),ndmin=2),axis=1)
        print('rupture')
        acc,cm = fitClassification(points, classValue1, classifier)
        for j in range(len(cmRupture)):
            cmRupture[j] += cm[j]
        classifierAccuracy1.append(acc) #Accuracy calculation
        if classifierAccuracy1[i] == 1:
            correctPrediction1 = correctPrediction1 + 1
        else:
            wrongPrediction1 = wrongPrediction1 + 1
        
        # if plotDiagRupture:    Cannot Do in this function
        #     addPoint(points,diagTrainScale1,trainClassifi1,'classifier',colors=colors,pointClass=classValueString1,classification_classes_set=trainClass,classification_classesName=className,figName=figName,xlabel=xlabel,ylabel=ylabel,save=save,folderPath=savePath)

    if plotDiagRupture:
        #Plotting accuracy bars
        plt.bar([0,1],height = [correctPrediction1,wrongPrediction1],width = 0.8)
        plt.xticks([0,1],['Correct prediction','Wrong prediction'])
        plt.title(classifierChoice + ' Stats set - Prediction accuracy for ' + str(len(testSerie)) + ' values divided in ' + str(len(classifierAccuracy1)) + 'sets' )
        plt.legend()
        saveFigure(save,savePath,'/' + classifierChoice + '_accuracy_pointSet_stats.png');
        plt.show()
        
    accuracyRupture = correctPrediction1/(wrongPrediction1+correctPrediction1)
    print('\nAccuracy Calculation Done!!!\n')
    
    
    
    #Test using all points of test set
    #Using classificationConfusionMatrixFunction
    print('\nConfusion matrix per point in progress...\n')  
    figName = classifierChoice + '_test_' + featureChoice + '_confusion_matrix'
    testCm1,testPredVerif1, = classificationConfusionMatrix(diagTestScale, realTestClass, classifier, [5,6],save=save,plot=plotDiagPerPoints,figName=figName,folderPath=savePath)
    figName = classifierChoice + '_test_' + featureChoice + '_points'
    accuracy = (testCm1[0][0]+testCm1[1][1])/(testCm1[0][0]+testCm1[1][1]+testCm1[0][1]+testCm1[1][0])
    accuracyPerPoint1 = accuracy
    cmPerPoint = [int(testCm1[0][0]),int(testCm1[0][1]),int(testCm1[1][0]),int(testCm1[1][1])]

    
    # if plotDiagPerPoints:     Cannot Do in this function
    #     addPoint(diagTestScale,diagTrainScale1,trainClassifi1,'classifier',colors=colors,pointClass='All set',classification_classes_set=trainClass,classification_classesName=className,figName=figName,xlabel=xlabel,ylabel=ylabel,save=save,folderPath=savePath)  


#accuracyPerPoint1 is the accuracy per points for each method listed in classificationList
#accuracyPerPoint2 is the confusion matrix per points for each method listed in classificationList
#accuracyRupture gives the accuracy per point for each method
    return accuracyPerPoint1,accuracyRupture,cmPerPoint, cmRupture


def doResultClassificationPerPoint(testSerie,diagTestScale,realTestClass,classifier,plot=0,xlabel='X',ylabel='Y',folderPath='',save=0,figName='resultClassification',savePath='',classifierChoice='classification',featureChoice=''):
    if plot == 0:
        plotDiagPerPoints = 0
    elif plot == 1 :
        plotDiagPerPoints = 1
    
    #Test using all points of test set
    #Using classificationConfusionMatrixFunction
    # print('\nConfusion matrix per point in progress...\n')  
    figName = classifierChoice + '_test_' + featureChoice + '_confusion_matrix'
    testCm1,testPredVerif1, = classificationConfusionMatrix(diagTestScale, realTestClass, classifier, [5,6],save=save,plot=plotDiagPerPoints,figName=figName,folderPath=savePath)
    figName = classifierChoice + '_test_' + featureChoice + '_points'
    accuracy = (testCm1[0][0]+testCm1[1][1])/(testCm1[0][0]+testCm1[1][1]+testCm1[0][1]+testCm1[1][0])
    accuracyPerPoint1 = accuracy
    cmPerPoint = [int(testCm1[0][0]),int(testCm1[0][1]),int(testCm1[1][0]),int(testCm1[1][1])]

    return accuracyPerPoint1,cmPerPoint


def doResultClassificationRupture(testSerie,diagTestScale,realTestClass,classifier,plot=0,xlabel='X',ylabel='Y',folderPath='',save=0,figName='resultClassification',savePath='',classifierChoice='classification',featureChoice=''):
    if plot == 0:
        plotTimeRupt = 0
        plotDiagRupture = 0
    elif plot == 1 :
        plotTimeRupt = 1
        plotDiagRupture = 1
    className = ['normal','Calculation','I/O consumption','Temperature variation','reset','latch','front de latch up']
    
    classifierAccuracy1 = []
    correctPrediction1 = 0
    wrongPrediction1 = 0
    cmRupture = [0,0,0,0]

            
    #Time Series Rupture
    timeRuptureAlgo,timeRuptureBreakPoints,timeRuptureDataBreak = timeRupture(testSerie[:,2],penaltyValue=5,folderPath=savePath,ylabel='Current',xlabel='Time (s)',save=save,plot=plotTimeRupt)

    
    #Test using rupture package        
    # print('\nAccuracy Calculation in progress...\n')
    for i in range(len(timeRuptureBreakPoints)-1):
        figName = classifierChoice + '_test_' + featureChoice + '_rupture_confusion_matrix_' + str(i+1) 
        
        indices1 = [timeRuptureBreakPoints[i],timeRuptureBreakPoints[i+1]]   
        classValue1 = getClass(indices1,realTestClass,className)   #getting the real class for the set of points
        points = np.array(diagTestScale[indices1,0].mean(),ndmin=2)
        for featureColumn in range(1,diagTestScale.shape[1]):
            points = np.append(points,np.array(diagTestScale[indices1,0].mean(),ndmin=2),axis=1)
        # print('rupture')
        acc,cm = fitClassification(points, classValue1, classifier)
        for j in range(len(cmRupture)):
            cmRupture[j] += cm[j]
        classifierAccuracy1.append(acc) #Accuracy calculation
        if classifierAccuracy1[i] == 1:
            correctPrediction1 = correctPrediction1 + 1
        else:
            wrongPrediction1 = wrongPrediction1 + 1
        
        # if plotDiagRupture:    Cannot Do in this function
        #     addPoint(points,diagTrainScale1,trainClassifi1,'classifier',colors=colors,pointClass=classValueString1,classification_classes_set=trainClass,classification_classesName=className,figName=figName,xlabel=xlabel,ylabel=ylabel,save=save,folderPath=savePath)

    if plotDiagRupture:
        #Plotting accuracy bars
        plt.bar([0,1],height = [correctPrediction1,wrongPrediction1],width = 0.8)
        plt.xticks([0,1],['Correct prediction','Wrong prediction'])
        plt.title(classifierChoice + ' Stats set - Prediction accuracy for ' + str(len(testSerie)) + ' values divided in ' + str(len(classifierAccuracy1)) + 'sets' )
        plt.legend()
        saveFigure(save,savePath,'/' + classifierChoice + '_accuracy_pointSet_stats.png');
        plt.show()
        
    accuracyRupture = correctPrediction1/(wrongPrediction1+correctPrediction1)
    # print('\nAccuracy Calculation Done!!!\n')
    

    return accuracyRupture, cmRupture


def ifacDataFrame (index = 'index0'):
    return pd.DataFrame(index = [index], 
                        columns=['test used', 'knn time', 'knn accuracy','knn k','knn weight','knn metric',
                             'svm time', 'svm accuracy','svm kernel','svm gamma','svm weigth',
                             'decision tree time', 'decision tree accuracy','decision tree criterion','decision tree splitter','decision tree min split',
                             'random forest time', 'random forest accuracy','random forest estimators','random forest criterion','random forest min split',
                             'kmeans time', 'kmeans accuracy','kmeans n cluster','kmeans init','kmeans n init', 'kmeans max iter',
                             'HC time', 'HC accuracy','HC n cluster','HC affinity','HC linkage',
                             'svm time', 'svm accuracy','svm epsilon','svm min sample','svm distance',
                             'dyclee time', 'dyclee accuracy','dyclee g_size','dyclee outlier rejection'])









def clusterScore(accuracy,Nclass,Ncluster,RatioPenalty,penalty):
    
    function = float(math.exp(Ncluster/Nclass - RatioPenalty))
    function = function ** (1.0/penalty)
    if Ncluster == 1:
        score = 0
    if Ncluster/Nclass <= RatioPenalty:
        score = accuracy
    else:
        score = float(accuracy/function)
    return score












