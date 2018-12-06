import numpy as np
from scipy.cluster import hierarchy
from scipy.spatial import distance_matrix
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets.samples_generator import make_blobs

'''
Generating Random Data
We will be generating a set of data using the make_blobs class. 

Input these parameters into make_blobs:

n_samples: The total number of points equally divided among clusters.
Choose a number from 10-1500
centers: The number of centers to generate, or the fixed center locations.
Choose arrays of x,y coordinates for generating the centers. Have 1-10 centers
(ex. centers=[[1,1], [2,5]])
cluster_std: The standard deviation of the clusters. The larger the number,
the further apart the clusters Choose a number between 0.5-1.5
'''
X1, y1 = make_blobs(n_samples=50, centers=[
                    [4, 4], [-2, -1], [1, 1], [10, 4]], cluster_std=0.9)
# Plot the scatter plot
plt.scatter(X1[:, 0], X1[:, 1], marker='o')

'''
Agglomerative Clustering
We will start by clustering the random data points we just created.

The Agglomerative Clustering class will require two inputs:

n_clusters: The number of clusters to form as well as the number of centroids
to generate.
Value will be: 4
linkage: Which linkage criterion to use. The linkage criterion determines
which distance to use between sets of observation. The algorithm will merge
the pairs of cluster that minimize this criterion.
Value will be: 'complete'
Note: It is recommended you try everything with 'average' as well
'''

agglom = AgglomerativeClustering(n_clusters=4, linkage='average')

# Fit the model with X2 and y2
agglom.fit(X1, y1)

# Create a figure of size 6 inches by 4 inches.
plt.figure(figsize=(6, 4))

# These two lines of code are used to scale the data points down,
# Or else the data points will be scattered very far apart.

# Create a minimum and maximum range of X1.
x_min, x_max = np.min(X1, axis=0), np.max(X1, axis=0)

# Get the average distance for X1.
X1 = (X1 - x_min) / (x_max - x_min)

# This loop displays all of the datapoints.
for i in range(X1.shape[0]):
    # Replace the data points with their respective cluster value
    # (ex. 0) and is color coded with a colormap (plt.cm.spectral)
    plt.text(X1[i, 0], X1[i, 1], str(y1[i]),
             color=plt.cm.nipy_spectral(agglom.labels_[i] / 10.),
             fontdict={'weight': 'bold', 'size': 9})

# Remove the x ticks, y ticks, x and y axis
plt.xticks([])
plt.yticks([])
# plt.axis('off')


# Display the plot of the original data before clustering
plt.scatter(X1[:, 0], X1[:, 1], marker='.')
# Display the plot
plt.show()


# distance matrix contains the distance from each point to
# every other point of a dataset .
dist_matrix = distance_matrix(X1, X1)
print(dist_matrix)

'''
Using the linkage class from hierarchy, pass in the parameters:

The distance matrix
'complete' for complete linkage
A Hierarchical clustering is typically visualized as a dendrogram as shown in
the following cell. Each merge is represented by a horizontal line.
The y-coordinate of the horizontal line is the similarity of the two clusters
that were merged, where cities are viewed as singleton clusters.
By moving up from the bottom layer to the top node, a dendrogram allows us
to reconstruct the history of merges that resulted in the depicted clustering.

Next, we will save the dendrogram to a variable called dendro.
In doing this, the dendrogram will also be displayed
'''
Z = hierarchy.linkage(dist_matrix, 'complete')
dendro = hierarchy.dendrogram(Z)


# Showing another clusters distance calculation method
Z1 = hierarchy.linkage(dist_matrix, 'average')
dendro1 = hierarchy.dendrogram(Z1)
