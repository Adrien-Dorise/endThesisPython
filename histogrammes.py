# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 17:53:58 2021

@author: addor
"""
import matplotlib.pyplot as plt
import numpy as np
%config InlineBackend.figure_format = 'retina'

bar_width = 0.35
opacity = 0.9
plt.style.use('ggplot')

classifAlgo = ['KNN', 'Naive Bayes','Deci Tree','Rand Forest','SVM']
classifAccPerPoint =  [97,96.5,94.5,95.5,97.8]
classifAccRupt = [94,94.5,92.7,94,95]

plt.plot(classifAccPerPoint)
plt.plot(classifAccRupt)
plt.grid()
plt.ylim([65,100])
plt.show()

n = len(classifAlgo)
fig, ax = plt.subplots()
index = np.arange(n)

ax.bar(index, classifAccPerPoint, bar_width, alpha=opacity, color='green', label='Sliding window')
ax.bar(index+bar_width, classifAccRupt, bar_width, alpha=opacity, color='b', label='Rupture window')
ax.set_xlabel('Classification algorithm')
ax.set_ylabel('Accuracy (%)')
ax.set_title('')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(classifAlgo)
ax.legend()
plt.ylim([60,100])
plt.grid(axis='x')
plt.show()


clusterAlgo = ['K-means', 'Hierar clust','DBSCAN','DyClee']
clusterAccPerPoint =  [89.35,91.27,65.4,89.28]
clusterAccRupt = [79.8,84.58,70,87.03]

plt.plot(clusterAccPerPoint)
plt.plot(clusterAccRupt)
plt.grid()
plt.ylim([60,100])
plt.show()

n = len(clusterAlgo)
fig, ax = plt.subplots()
index = np.arange(n)

ax.bar(index, clusterAccPerPoint, bar_width, alpha=opacity, color='green', label='Sliding time window')
ax.bar(index+bar_width, clusterAccRupt, bar_width, alpha=opacity, color='b', label='Rupture time window')
ax.set_xlabel('Clustering algorithm')
ax.set_ylabel('Accuracy (%)')
ax.set_title('')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(clusterAlgo)
ax.legend()
plt.ylim([60,100])
plt.grid(axis='x')
plt.show()

