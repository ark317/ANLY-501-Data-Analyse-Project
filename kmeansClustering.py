# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 15:34:39 2017

We will perform the kmeans clustering on Zillow_Cleaned.csv

@author: Armaan Khullar
"""


#hierarchial clustering (ward)
#k-means
#dbscan

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

from sklearn.cluster import KMeans
from sklearn import preprocessing
import pylab as pl
from sklearn import decomposition
from pprint import pprint
from sklearn.metrics import silhouette_samples, silhouette_score

def kmeans():
    myData = pd.read_csv('Zillow_Cleaned.csv' , sep=',', encoding='latin1')
    output = open("kmeans_output.txt", "w")
    x= myData.iloc[:,2:19].values
    
    #x = myData.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    normalizedDataFrame = pd.DataFrame(x_scaled)
    
    best_silhouette = 0 #will find the best silhouette score among all the K's
    best_k = 0    #will store the K that has the best silhouette score
    
    for k in [3, 4, 5, 6, 7, 8, 9, 10]: #We now perform the K-means analysis for values of K = 3, 4, and 10.
        output.write("\n")
        output.write("K= " + str(k) +"\n") #Write the value of K
        print("K= ", str(k))
        kmeans = KMeans(n_clusters=k)   #Cluster the data
        cluster_labels = kmeans.fit_predict(normalizedDataFrame) #get cluster labels
        
        # Determine if the clustering is good
        silhouette_avg = silhouette_score(normalizedDataFrame, cluster_labels) #Get the silhouette score
        if silhouette_avg > best_silhouette:
            best_silhouette = silhouette_avg
            best_K = k
            
        output.write("For n_clusters = "+ str(k)+ " The average silhouette_score is: "+ str(silhouette_avg) +"\n")
        print("For n_clusters =", str(k)," the average silhouette_score is:", str(silhouette_avg),"\n")
    
        centroids = kmeans.cluster_centers_      #Get the centroids of the clusters  
        
        #pprint(cluster_labels, output) #printing out the cluster labels to file
        #pprint(cluster_labels)
        #pprint(centroids, output) #writing the centroids to the output file
        #pprint(centroids)
        #pprint(prediction)
        
        
        #####
        # PCA
        # Let's convert our high dimensional data to 2 dimensions
        # using PCA
        pca2D = decomposition.PCA(2)
        
        # Turn the NY Times data into two columns with PCA
        plot_columns = pca2D.fit_transform(normalizedDataFrame)
        
        # Plot using a scatter plot and shade by cluster label
        plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=cluster_labels)
        plt.show() #Display the plot
    
    output.close()  #close the output file for this function
    
    print("The best silhouette score is:", best_silhouette, "at K =", best_K)
    
kmeans()





