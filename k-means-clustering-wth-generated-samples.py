'''
K-means is vastly used for clustering in many data science applications,
especially useful if you need to quickly discover insights from unlabeled data.
It is only guaranteed to converge to local optima
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs

# Create random seed, a parameter that determines the sequence of
# generated pseudorandom numbers
# clustering result will depend on the initialization chosen,
# and hence depend on the seed.
# By fixing the seed, you ensure that the ‘random’ numbers generated in ML alg
# are exactly the same, every time it is ran.
# This means your experiments/results will be exactly reproducible
np.random.seed(0)

# making random clusters of points by using the make_blobs class.
# The make_blobs class can take in many inputs,
# but we will be using these specific ones.

# Input
# n_samples: The total number of points equally divided among clusters.
# Value will be: 5000
# centers: The number of centers to generate, or the fixed center locations.
# Value will be: [[4, 4], [-2, -1], [2, -3],[1,1]]
# cluster_std: The standard deviation of the clusters.
# Value will be: 0.9

# Output
# X: Array of shape [n_samples, n_features]. (Feature Matrix)
# The generated samples.
# y: Array of shape [n_samples]. (Response Vector)
# The integer labels for cluster membership of each sample.
X, y = make_blobs(n_samples=5000, centers=[
                  [4, 4], [-2, -1], [2, -3], [1, 1]], cluster_std=0.9)
plt.scatter(X[:, 0], X[:, 1], marker='.')
plt.show()

# Setting up K-Means
# The KMeans class has many parameters that can be used,
# but we will be using these three:

# init: Initialization method of the centroids.
# Value will be: "k-means++"
# k-means++: Selects initial cluster centers for k-mean clustering in a smart
# way to speed up convergence.
# n_clusters: The number of clusters to form as well as the number of
# centroids to generate.
# Value will be: 4 (since we have 4 centers)
# n_init: Number of time the k-means algorithm will be run with different
# centroid seeds. The final results will be the best output of n_init
# consecutive runs in terms of inertia.
# Value will be: 12
k_means = KMeans(init="k-means++", n_clusters=4, n_init=12)
# Fit the KMeans model with the feature matrix we created above, X
k_means.fit(X)

# grab the labels for each point in the model using KMeans'
# .labels_attribute & save it as k_means_labels
k_means_labels = k_means.labels_
print(k_means_labels)

# get the coordinates of the cluster centers using KMeans'
# .cluster_centers_ & save it as k_means_cluster_centers
k_means_cluster_centers = k_means.cluster_centers_
print(k_means_cluster_centers)

# Creating the Visual Plot
# plot them and see what it looks like!

# Initialize the plot with the specified dimensions.
fig = plt.figure(figsize=(6, 4))

# Colors uses a color map, which will produce an array of colors based on
# the number of labels there are. We use set(k_means_labels) to get the
# unique labels.
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))

# Create a plot
ax = fig.add_subplot(1, 1, 1)

# For loop that plots the data points and centroids.
# k will range from 0-3, which will match the possible clusters that each
# data point is in.
for k, col in zip(range(len([[4, 4], [-2, -1], [2, -3], [1, 1]])), colors):

    # Create a list of all data points, where the data points that are
    # in the cluster (ex. cluster 0) are labeled as true, else they are
    # labeled as false.
    my_members = (k_means_labels == k)

    # Define the centroid, or cluster center.
    cluster_center = k_means_cluster_centers[k]

    # Plots the datapoints with color col.
    ax.plot(X[my_members, 0], X[my_members, 1],
            'w', markerfacecolor=col, marker='.')

    # Plots the centroids with specified color, but with a darker outline
    ax.plot(cluster_center[0], cluster_center[1], 'o',
            markerfacecolor=col,  markeredgecolor='k', markersize=6)

# Title of the plot
ax.set_title('KMeans')

# Remove x-axis ticks
ax.set_xticks(())

# Remove y-axis ticks
ax.set_yticks(())

# Show the plot
plt.show()

# Same as above, but with 3 clusters instead of 4
k_means3 = KMeans(init="k-means++", n_clusters=3, n_init=12)
k_means3.fit(X)
fig = plt.figure(figsize=(6, 4))
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means3.labels_))))
ax = fig.add_subplot(1, 1, 1)
for k, col in zip(range(len(k_means3.cluster_centers_)), colors):
    my_members = (k_means3.labels_ == k)
    cluster_center = k_means3.cluster_centers_[k]
    ax.plot(X[my_members, 0], X[my_members, 1],
            'w', markerfacecolor=col, marker='.')
    ax.plot(cluster_center[0], cluster_center[1], 'o',
            markerfacecolor=col,  markeredgecolor='k', markersize=6)
# Title of the plot
ax.set_title('KMeans with 3 clusters')

# Remove x-axis ticks
ax.set_xticks(())

# Remove y-axis ticks
ax.set_yticks(())

# Show the plot
plt.show()
