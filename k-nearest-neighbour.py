'''
K-Nearest Neighbors is an algorithm for supervised learning.
Where the data is 'trained' with data points
corresponding to their classification.
Once a point is to be predicted, it takes into account the
'K' nearest points to it to determine it's classification.
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

'''
Dataset description
Telecommunications provider has segmented its customer base by
service usage patterns, categorizing the customers into four groups.
If demographic data can be used to predict group membership,
the company can customize offers for individual prospective customers.
It is a classification problem. That is, given the dataset,
with predefined labels, we need to build a model to be used
to predict class of a new or unknown case.
'''

# Reading the data in
df = pd.read_csv("teleCust1000t.csv")

# how many of each class is in our data set
# print(df['custcat'].value_counts())

# Explore your data using visualization techniques
df.hist(column='income', bins=50)

# To use SciKit Learn, convert pandas to numpy array
X = df[['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed',
        'employ', 'retire', 'gender', 'reside']] .values

# Our labels
y = df['custcat'].values

# Data Standardization give data zero mean and unit variance,
# it is good practice, especially for algorithms such as KNN
# which is based on distance of cases
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
X[0:5]

# Doing a train and test on the same dataset will most likely have
# low out-of-sample accuracy, due to the likelihood of being over-fit.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=4)
print('Train set:', X_train.shape,  y_train.shape)
print('Test set:', X_test.shape,  y_test.shape)


# Train Model and Predict
k = 4
neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
# Show model hyper parameters
# print(neigh)

# use model to predict
yhat = neigh.predict(X_test)

# Evaluation Metrics
print("Train set Accuracy: ", metrics.accuracy_score(
    y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))


'''
How to Choose the best K value?
'''

Ks = 20
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
ConfustionMx = []
for n in range(1, Ks):
    # Train Model and Predict
    neigh = KNeighborsClassifier(
        n_neighbors=n, n_jobs=-1).fit(X_train, y_train)
    yhat = neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)
    std_acc[n-1] = np.std(yhat == y_test)/np.sqrt(yhat.shape[0])

mean_acc

plt.plot(range(1, Ks), mean_acc, 'g')
plt.fill_between(range(1, Ks), mean_acc - 1 * std_acc,
                 mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show()

print("The best accuracy was with", mean_acc.max(),
      "with k=", mean_acc.argmax()+1)
