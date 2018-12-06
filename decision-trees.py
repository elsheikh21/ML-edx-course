'''
machine learning algorithm, Decision Tree. 
Will use this classification algorithm to build a model from historical data
of patients, and their response to different medications.
Then you use the trained decision tree to predict the class of an
unknown patient, or to find a proper drug for a new patient.
'''

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn import tree
import matplotlib.image as mpimg
import pydotplus
from sklearn.externals.six import StringIO
import matplotlib.pyplot as plt

# The feature sets of this dataset {Age, Sex, Blood Pressure, Cholesterol}
# and the target is the drug, that each patient responded to.

# read data using pandas data-frame
my_data = pd.read_csv("drug200.csv", delimiter=",")

# size of data
print(my_data.size)

# Pre-processing Step
# Remove the column containing the target name
# since it doesn't contain numeric values
X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values

le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F', 'M'])
X[:, 1] = le_sex.transform(X[:, 1])


le_BP = preprocessing.LabelEncoder()
le_BP.fit(['LOW', 'NORMAL', 'HIGH'])
X[:, 2] = le_BP.transform(X[:, 2])


le_Chol = preprocessing.LabelEncoder()
le_Chol.fit(['NORMAL', 'HIGH'])
X[:, 3] = le_Chol.transform(X[:, 3])

print(X[0:5])

y = my_data["Drug"]
print(y[0:5])

# train/test split on our decision tree
X_trainset, X_testset, y_trainset, y_testset = train_test_split(
    X, y, test_size=0.3, random_state=3)

# Print shape of training and testing data
print(X_trainset.shape)
print(y_trainset.shape)

print(X_testset.shape)
print(y_testset.shape)

# first create an instance of the DecisionTreeClassifier called drugTree.
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth=4)
# it shows the default parameters
print(drugTree)
# train the model
drugTree.fit(X_trainset, y_trainset)
# predict
predTree = drugTree.predict(X_testset)
# print first 5 entries of the output
print(predTree[0:5])
print(y_testset[0:5])

accuracy = metrics.accuracy_score(y_testset, predTree)
# Evaluate performance of the decision trees
print("DecisionTrees's Accuracy: ", accuracy)

acc = np.mean(y_testset == predTree)
print("DecisionTrees's Accuracy without SciKit Learn ", acc)

# visualize the tree
dot_data = StringIO()
filename = "drugtree.png"
featureNames = my_data.columns[0:5]
targetNames = my_data["Drug"].unique().tolist()
out = tree.export_graphviz(drugTree, feature_names=featureNames,
                           out_file=dot_data, class_names=np.unique(
                               y_trainset), filled=True,
                           special_characters=True, rotate=False)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img, interpolation='nearest')
