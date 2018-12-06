import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from sklearn.datasets.samples_generator import make_blobs

# Load csv & visualize it
cust_df = pd.read_csv("Cust_Segmentation.csv")
print(cust_df.head())

# Preprocess it
# Address in this dataset is a categorical variable. k-means algorithm isn't
# directly applicable to categorical variables because Euclidean distance
# function isn't really meaningful for discrete variables.
# Cut it off & check data one more time
df = cust_df.drop('Address', axis=1)
print(df.head())

# Normalize Data
# It is a statistical method that helps mathematical-based algorithms
# to interpret features with different magnitudes & distributions equally
X = df.values[:, 1:]
X = np.nan_to_num(X)
Clus_dataSet = StandardScaler().fit_transform(X)
print(Clus_dataSet)

# Lets apply k-means on our dataset & take look at cluster labels.
clusterNum = 3
k_means = KMeans(init="k-means++", n_clusters=clusterNum, n_init=12)
k_means.fit(X)
labels = k_means.labels_
print(labels)

# Assign the labels to each row in data-frame.
df["Clus_km"] = labels
print(df.head(5))

# easily check the centroid values by averaging the features in each cluster.
print(df.groupby('Clus_km').mean())

# Look at the distribution of customers based on their age and income
area = np.pi * (X[:, 1])**2
plt.scatter(X[:, 0], X[:, 3], s=area, c=labels.astype(np.float), alpha=0.5)
plt.xlabel('Age', fontsize=16)
plt.ylabel('Income', fontsize=16)
plt.show()

# Visualize it on 3D plot
fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
ax.set_xlabel('Education')
ax.set_ylabel('Age')
ax.set_zlabel('Income')

ax.scatter(X[:, 1], X[:, 0], X[:, 3], c=labels.astype(np.float))
plt.show()
