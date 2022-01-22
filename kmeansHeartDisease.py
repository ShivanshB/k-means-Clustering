import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Reading the CSV data file from UCI School of Medicine.
# Parsing the data into numpy arrays to perform clustering on.
data = pd.read_csv('heartDiseaseUCI.csv')
dataColumns = pd.DataFrame(data, columns= ['thalach','age'])
X = dataColumns.to_numpy()


# Marks every point in the input array with its closest clustering center.
def markPoints(X, K):
    for i in range(X.shape[0]):
        closestCluster = 0
        min = float('inf')
        for k in range(K.shape[0]):
            # print(k)
            val = abs((K[k,0] - X[i,0])**2 + (K[k,1] - X[i,1])**2)
            if val < min:
                # print("new closest ", k)
                closestCluster = k
                min = val
        X[i,2] = closestCluster

# Updates cluster centers based on the average of the data points in the cluster
def updateClusterCenters(X, K):
    a = np.mean(X[X[:,2]==0],axis = 0)[:2]
    b = np.mean(X[X[:,2]==1],axis = 0)[:2]
    c = np.mean(X[X[:,2]==2],axis = 0)[:2]
    K = np.array([a,b,c])


# Performs the clustering and data displaying while calling the other helper functions
def kCluster(X, K):

    # Initialize cluster centers and scale by an approximate factor of 100 to speed up convergence.
    initialK = np.array([[125,64],[120,42],[152,60]])

    # Add column of -1 to dataset as placeholders for closest cluster value (0 --> (K-1))
    X = np.c_[X,np.full(X.shape[0],-1)]

    # Clusters and recalculates centers 100 times
    for i in range(1000):
        markPoints(X, initialK)
        updateClusterCenters(X, initialK)

    # Separates data into each cluster and plots it by color.
    a = X[X[:,2] == 0]
    b = X[X[:,2] == 1]
    c = X[X[:,2] == 2]

    plt.scatter(a[:,0],a[:,1], color = "yellow")
    plt.scatter(b[:,0],b[:,1], color = "orange")
    plt.scatter(c[:,0],c[:,1], color = "lightgreen")

    K = initialK

    # Plots cluster centers with black X's
    plt.scatter(K[0,0],K[0,1], marker = 'x', color = "black")
    plt.scatter(K[1,0],K[1,1], marker = 'x', color = "black")
    plt.scatter(K[2,0],K[2,1], marker = 'x', color = "black")
   
    # Labels the graph and scales appropriately
    plt.axis([0, 200, 0, 100])
    plt.xlabel("Maximum Achieved Heart Rate")
    plt.ylabel("Age (years)")
    plt.title("K-means Clustering of Heart Disease Patients \n Using Data From UCI School of Medicine")
    plt.show()

kCluster(X,3)
