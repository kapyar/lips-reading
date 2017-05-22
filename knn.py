# -*- coding: utf-8 -*-
# USAGE
# python knn.py --data data/aspect.dat


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn.cluster import KMeans


x = [1, 2, 5, 7, 8]
y = [1, 2, 2, 3, 9]


X = np.array([
    [1, 1],
    [3, 3],

])

kmeans = KMeans(n_clusters=1)
kmeans.fit(X)


centroids = kmeans.cluster_centers_
print ("centroids {}".format(centroids))


plt.scatter(centroids[0][0], centroids[0][1])
plt.show()