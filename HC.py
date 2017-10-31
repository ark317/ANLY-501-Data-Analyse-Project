# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 15:53:35 2017

@author: Armaan Khullar
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import LabelEncoder
import scipy.cluster.hierarchy as sch 
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn import preprocessing
import pylab as pl
from sklearn import decomposition
from pprint import pprint


def HC():
    myData = pd.read_csv('graduation_rates_CLEANED.csv')
    
    print("Before doing the relabeling:")
    X = myData.iloc[:, [4,5,6,7, 8, 9, 10]].values
    print(X)
    
    print("\nNow, we are doing the relabeling:")
    lb = LabelEncoder()
    myData["State"] = lb.fit_transform(myData["State"])
    myData["City"] = lb.fit_transform(myData["City"])

    
    X = myData.iloc[:, [4,5,6,7, 8, 9, 10]].values
    print(X)
    
    #min_max_scaler = preprocessing.MinMaxScaler()
    #x_scaled = min_max_scaler.fit_transform(X)
    #normalizedDataFrame = pd.DataFrame(x_scaled)
    #now using the dendogram to find the best number of clusters.
    
    dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
    plt.title('Dendogram')
    plt.xlabel('Graduation')
    plt.ylabel('Euclidean distances')
    plt.show()
    
    #It seems like there are 4 clusters according to the dendrogram.
    #we now fit the hierarchial clustering to the graduation dataset
    hc = AgglomerativeClustering(n_clusters = 4, affinity = "euclidean", linkage = "ward")
    cluster_labels = hc.fit_predict(X) #getting the cluster labels
    
    silhouette_avg = silhouette_score(X, cluster_labels) #Get the silhouette score
    print("The silhouette score is:",str(silhouette_avg))
        
    #####
    # PCA
    # Let's convert our high dimensional data to 2 dimensions
    # using PCA
    print("\nWe are now using PCA:")
    pca2D = decomposition.PCA(2)
        
    # Turn the NY Times data into two columns with PCA
    plot_columns = pca2D.fit_transform(X)
        
    # Plot using a scatter plot and shade by cluster label
    plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=cluster_labels)
    plt.show() #Display the plot
    

HC()   