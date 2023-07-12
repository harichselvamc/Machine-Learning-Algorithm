# Example: Cluster (group) data points
# (In ML a group is called "cluster")
#
# Given data points, algorithm divides into groups.
# Run the program. You see the "purple group" and "yellow group". 
#
# Tasks:
# * Find groups in set of points
# * Given new data point, predict group
#
# Clustering does not have a training phase, that's why its 
# "unsupervised learning". KMeans is a clustering algorithm.
#

from sklearn.cluster import KMeans
import numpy as np
#from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt

# Data Points. Just (X,Y) pairs.
X = np.array([[-3.79993965,  4.92713237],
              [-2.46423863,  4.12236291],
              [ 3.89987334,  7.85705931],
              [-3.11523096,  3.77245361],
              [-3.48623175,  3.8148719 ],
              [ 4.59866351,  8.00671429],
              [ 4.79297733,  7.15651578],
              [-1.6230895,   3.86299429],
              [-2.88428411,  4.89272643],
              [-2.85958091,  4.38366608],
              [-3.25812003,  4.70782312],
              [-3.63688602,  3.8682572 ],
              [-2.92153638,  4.49152321],
              [ 4.77470279,  6.9536221 ],
              [ 4.83124038,  7.28170544],
              [-3.42949007,  3.57349703],
              [ 4.26687167,  6.81163081],
              [ 4.46741295,  7.83138553],
              [-2.07464308,  4.27356787],
              [ 3.92774122,  7.25477767],
              [ 5.11612007,  8.07679629],
              [ 3.91297502,  7.64022513],
              [ 4.07707651,  7.56832056],
              [-2.52601024,  4.42793143],
              [ 4.4327831,  7.11303077],
              [ 4.46056019,  7.55039758],
              [-2.72107294,  4.58413728],
              [ 4.35909039,  6.36596775],
              [-2.83262742,  4.04997674],
              [ 4.16395926,  7.21307477],
              [-3.79630184,  3.86958788],
              [ 4.28465207,  8.42653829],
              [-2.93913617,  4.64097164],
              [ 4.65509449,  7.55151564],
              [-2.96653274,  5.08738576]])

# Initialize clustering algorithm named "kmeans" with "2 clusters".
kmeans = KMeans(n_clusters=2).fit(X)

# Given coordinate, which cluster does it belong to? 
print( kmeans.predict([[4, 1]]) )

# Visualize what we are doing
#-------------------------------
# Show data points
plt.scatter(X[:, 0], X[:, 1], s=50)

# Show clusters
y_kmeans = kmeans.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis_r')

# Show cluster centers
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.show()
