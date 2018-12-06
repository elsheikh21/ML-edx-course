
'''
Linear Regression is suited for estimating continuous values
(e.g. estimating house price),
it isn't the best tool for predicting the class of an observed data point.
In order to estimate the class of a data point, we need some sort of guidance
on what would be the most probable class for that data point.
For this, we use Logistic Regression.

Logistic Regression is a variation of Linear Regression,
useful when the observed dependent variable, y, is categorical.
It produces a formula that predicts the probability of the class label
as a function of the independent variables.

Logistic regression fits a special s-shaped curve by taking the linear
regression and transforming the numeric estimate into a probability with the
following function, which is called sigmoid function 𝜎
'''
import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import itertools
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import log_loss
'''
Telecommunications dataset for predicting customer churn.
This is a historical customer dataset where each row represents one customer.

The dataset includes information about:

Customers who left within the last month – the column is called Churn
Services that each customer has signed up for
Customer account information – how long they’ve been a customer
Demographic info about customers – gender, age range
'''

# Load data from CSV
churn_df = pd.read_csv("ChurnData.csv")

# Preprocessing and choosing data to model
churn_df = churn_df[['tenure', 'age', 'address', 'income',
                     'ed', 'employ', 'equip', 'callcard', 'wireless', 'churn']]
churn_df['churn'] = churn_df['churn'].astype('int')

# How many rows and cols in dataset
print(churn_df.shape)

# Name of cols in dataset
print(list(churn_df.columns.values))

# Define X and Y
X = np.asarray(churn_df[['tenure', 'age', 'address',
                         'income', 'ed', 'employ', 'equip']])
y = np.asarray(churn_df['churn'])

# Normalize dataset, help convergence of the technique used for optimization
X = preprocessing.StandardScaler().fit(X).transform(X)

# Train\Test split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=4)
print('Train set:', X_train.shape,  y_train.shape)
print('Test set:', X_test.shape,  y_test.shape)

# Modeling Logistic Regression
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train, y_train)
print(LR)

# Predicting
yhat = LR.predict(X_test)

# Predicting probabilities
# predict_proba: returns estimates for all classes,
# ordered by the label of classes.
yhat_prob = LR.predict_proba(X_test)

# Evaluation metrices
# Jaccard Index
print("Jaccard Index is: ", jaccard_similarity_score(y_test, yhat))

# Confusion Matrix


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


print(confusion_matrix(y_test, yhat, labels=[1, 0]))


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[1, 0])
np.set_printoptions(precision=2)


# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=[
                      'churn=1', 'churn=0'], normalize=False,
                      title='Confusion matrix')

print(classification_report(y_test, yhat))

# Log loss
print(log_loss(y_test, yhat_prob))

'''
Checking effects of different solvers, as well as
different regularization variable C
'''

# Modeling Logistic Regression
LR = LogisticRegression(C=0.01, solver='sag').fit(X_train, y_train)
# Predicting
yhat = LR.predict(X_test)
# Predicting probabilities
yhat_prob = LR.predict_proba(X_test)
# Log loss
print(log_loss(y_test, yhat_prob))

# Modeling Logistic Regression
LR = LogisticRegression(C=1.0, solver='sag').fit(X_train, y_train)
# Predicting
yhat = LR.predict(X_test)
# Predicting probabilities
yhat_prob = LR.predict_proba(X_test)
# Log loss
print(log_loss(y_test, yhat_prob))
