import matplotlib.cm as cm
from scipy.cluster.hierarchy import fcluster
import scipy.cluster.hierarchy
import pylab
import scipy
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial import distance_matrix
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering


# Read csv
pdf = pd.read_csv('cars_clus.csv')
print("Shape of dataset: ", pdf.shape)
pdf.head(5)

# Clean data
print("Shape of dataset before cleaning: ", pdf.size)
pdf[['sales', 'resale', 'type', 'price', 'engine_s',
     'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
     'mpg', 'lnsales']] = pdf[['sales', 'resale', 'type', 'price', 'engine_s',
                               'horsepow', 'wheelbas', 'width', 'length',
                               'curb_wgt', 'fuel_cap', 'mpg', 'lnsales']].apply(pd.to_numeric, errors='coerce')
pdf = pdf.dropna()
pdf = pdf.reset_index(drop=True)
print("Shape of dataset after cleaning: ", pdf.size)
pdf.head(5)

# Feature Selection
featureset = pdf[['engine_s',  'horsepow', 'wheelbas',
                  'width', 'length', 'curb_wgt', 'fuel_cap', 'mpg']]

# Normalization
x = featureset.values  # returns a numpy array
min_max_scaler = MinMaxScaler()
feature_mtx = min_max_scaler.fit_transform(x)
feature_mtx[0:5]

# Clustering using Scipy
leng = feature_mtx.shape[0]
D = scipy.zeros([leng, leng])
for i in range(leng):
    for j in range(leng):
        D[i, j] = scipy.spatial.distance.euclidean(
            feature_mtx[i], feature_mtx[j])

'''
agglomerative clustering, at each iteration,
the algorithm must update the distance matrix to reflect the distance
of the newly formed cluster with the remaining clusters in the forest.
The following methods are supported in Scipy for calculating the distance
between the newly formed cluster and each:

- single
- complete
- average
- weighted
- centroid
'''

Z = hierarchy.linkage(D, 'complete')

# Hierarchical clustering does not require a pre-specified number of clusters.
# However, in some applications we want a partition of disjoint clusters just
# as in flat clustering.
max_d = 3
clusters = fcluster(Z, max_d, criterion='distance')
print(clusters)


# can determine the number of clusters directly
k = 5
clusters = fcluster(Z, k, criterion='maxclust')
print(clusters)

# plot the dendrogram
fig = pylab.figure(figsize=(18, 50))


def llf(id):
    return '[%s %s %s]' % (pdf['manufact'][id], pdf['model'][id],
                           int(float(pdf['type'][id])))


dendro = hierarchy.dendrogram(
    Z,  leaf_label_func=llf, leaf_rotation=0,
    leaf_font_size=12, orientation='right')

# Clustering using scikit-learn
dist_matrix = distance_matrix(feature_mtx, feature_mtx)
print(dist_matrix)

'''
'AgglomerativeClustering' function from scikit-learn library to cluster the
dataset. The AgglomerativeClustering performs a hierarchical clustering using
a bottom up approach. The linkage criteria determines the metric used for
the merge strategy:

Ward minimizes the sum of squared differences within all clusters.
It is a variance-minimizing approach and in this sense is similar to the
k-means objective function but tackled with an agglomerative hierarchical
approach.
Maximum or complete linkage minimizes the maximum distance between
observations of pairs of clusters.
Average linkage minimizes the average of the distances between all
observations of pairs of clusters.
'''

agglom = AgglomerativeClustering(n_clusters=6, linkage='complete')
agglom.fit(feature_mtx)
agglom.labels_

# add a new field to our data-frame to show the cluster of each row
pdf['cluster_'] = agglom.labels_
pdf.head()

n_clusters = max(agglom.labels_)+1
colors = cm.rainbow(np.linspace(0, 1, n_clusters))
cluster_labels = list(range(0, n_clusters))

# Create a figure of size 6 inches by 4 inches.
plt.figure(figsize=(16, 14))

for color, label in zip(colors, cluster_labels):
    subset = pdf[pdf.cluster_ == label]
    for i in subset.index:
        plt.text(subset.horsepow[i], subset.mpg[i],
                 str(subset['model'][i]), rotation=25)
    plt.scatter(subset.horsepow, subset.mpg, s=subset.price*10,
                c=color, label='cluster'+str(label), alpha=0.5)
#    plt.scatter(subset.horsepow, subset.mpg)
plt.legend()
plt.title('Clusters')
plt.xlabel('horsepow')
plt.ylabel('mpg')

# Use them to distinguish the classes, and summarize the cluster.
# First we count the number of cases in each group
pdf.groupby(['cluster_', 'type'])['cluster_'].count()

agg_cars = pdf.groupby(['cluster_', 'type'])[
    'horsepow', 'engine_s', 'mpg', 'price'].mean()
agg_cars

# Hierarchical clustering could forge the clusters and discriminate
# them with quite high accuracy
plt.figure(figsize=(16, 10))
for color, label in zip(colors, cluster_labels):
    subset = agg_cars.loc[(label,), ]
    for i in subset.index:
        plt.text(subset.loc[i][0]+5, subset.loc[i][2], 'type=' +
                 str(int(i)) + ', price='+str(int(subset.loc[i][3]))+'k')
    plt.scatter(subset.horsepow, subset.mpg, s=subset.price *
                20, c=color, label='cluster'+str(label))
plt.legend()
plt.title('Clusters')
plt.xlabel('horsepow')
plt.ylabel('mpg')
