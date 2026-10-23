import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv(r"C:\Users\raymo\NareshIT_DataScience_AI\Machine_Learning_Projects\Clustering\K-Means_clustering\Mall_Customers.csv")


x = dataset.iloc[:, [3,4]].values 


import scipy.cluster.hierarchy as sch


dendogram = sch.dendrogram(sch.linkage(x,method='ward'))

plt.tittle('Dendogram')
plt.xlabel('Customers')
plt.ylabel('Euclidian Distance')
plt.show()


# Training the hierarchical clustering model

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, metric='euclidean', linkage='ward')
y_hc = hc.fit_predict(x)
print(y_hc)




# Visualizing the clusters

plt.scatter(x[y_hc == 0, 0], x[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(x[y_hc == 1, 0], x[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(x[y_hc == 2, 0], x[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(x[y_hc == 3, 0], x[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(x[y_hc == 4, 0], x[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')

plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

dataset['cluster'] = y_hc




