---
title: "Image classification, dimensionality reduction"
date: 2019-02-20
tags: [data science, data mining, machine learning, random forest, predictions]
header:
  image: "images/background.jpg"
excerpt: "data science, econometrics, regression"
---

``` python
%matplotlib inline

#Array processing
import numpy as np

#Data analysis, wrangling and common exploratory operations
import pandas as pd
from pandas import Series, DataFrame

#For visualization. Matplotlib for basic viz and seaborn for more stylish figures + statistical figures not in MPL.
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.core.display import Image

from sklearn.datasets import fetch_mldata
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler

import pydot, io
import time

#######################End imports###################################

try:
    mnist = fetch_mldata("MNIST original")

except Exception as ex:
    import tensorflow.examples.tutorials.mnist.input_data as input_data
    m=input_data.read_data_sets("MNIST")
    data = np.concatenate((m.train.images, m.test.images))
    target = np.concatenate((m.train.labels, m.test.labels))
    class dataFrame:
        def __init__(self, data, target):
            self.data = data
            self.target = target
    mnist = dataFrame(data, target)

#The data is organized as follows:
#  Each row corresponds to an image
#  Each image has 28*28 pixels which is then linearized to a vector of size 784 (ie. 28*28)
# mnist.data gives the image information while mnist.target gives the number in the image
print("#Images = %d and #Pixel per image = %s" % (mnist.data.shape[0], mnist.data.shape[1]))

#Print first row of the dataset
img = mnist.data[0]
print("First image shows %d" % (mnist.target[0]))
print("The corresponding matrix version of image is \n" , img)
print("The image in grey shape is ")
plt.imshow(img.reshape(28, 28), cmap="Greys")

#First 60K images are for training and last 10K are for testing
all_train_data = mnist.data[:60000]
all_test_data = mnist.data[60000:]
all_train_labels = mnist.target[:60000]
all_test_labels = mnist.target[60000:]


#For the first task, we will be doing binary classification and focus  on two pairs of
#  numbers: 7 and 9 which are known to be hard to distinguish
#Get all the seven images
sevens_data = mnist.data[mnist.target==7]
#Get all the none images
nines_data = mnist.data[mnist.target==9]
#Merge them to create a new dataset
binary_class_data = np.vstack([sevens_data, nines_data])
binary_class_labels = np.hstack([np.repeat(7, sevens_data.shape[0]), np.repeat(9, nines_data.shape[0])])

#In order to make the experiments repeatable, we will seed the random number generator to a known value
# That way the results of the experiments will always be same
np.random.seed(1234)
#randomly shuffle the data
binary_class_data, binary_class_labels = shuffle(binary_class_data, binary_class_labels)
print("Shape of data and labels are :" , binary_class_data.shape, binary_class_labels.shape)

#There are approximately 14K images of 7 and 9.
#Let us take the first 5000 as training and remaining as test data
orig_binary_class_training_data = binary_class_data[:5000]
binary_class_training_labels = binary_class_labels[:5000]
orig_binary_class_testing_data = binary_class_data[5000:]
binary_class_testing_labels = binary_class_labels[5000:]

#The images are in grey scale where each number is between 0 to 255
# Now let us normalize them so that the values are between 0 and 1.
# This will be the only modification we will make to the image
binary_class_training_data = orig_binary_class_training_data / 255.0
binary_class_testing_data = orig_binary_class_testing_data / 255.0
scaled_training_data = all_train_data / 255.0
scaled_testing_data = all_test_data / 255.0

print(binary_class_training_data[0,:])

def plot_dtree(model,fileName):
    #You would have to install a Python package pydot
    #You would also have to install graphviz for your system - see http://www.graphviz.org/Download..php
    #If you get any pydot error, see url
    # http://stackoverflow.com/questions/15951748/pydot-and-graphviz-error-couldnt-import-dot-parser-loading-of-dot-files-will
    dot_tree_data = io.StringIO()
    tree.export_graphviz(model, out_file = dot_tree_data)
    (dtree_graph,) = pydot.graph_from_dot_data(dot_tree_data.getvalue())
    dtree_graph.write_png(fileName)

```

``` python

# I first assign variable clf to an instantiated Decision Tree Classifier
# with parameters (criterion=entropy and  random state=(1234))
clf = tree.DecisionTreeClassifier(criterion="entropy", random_state=(1234))

# I train a decision tree on normalised training data and training labels
fitting = clf.fit(binary_class_training_data,binary_class_training_labels )

# Based on built decision tree fitted in previous point, predict the labels of normalised testing data
predictions = clf.predict(binary_class_testing_data)

# Calling function accuracy_score from imported metrics library
print("The accuracy of Decision Tree classifier is", round(metrics.accuracy_score(binary_class_testing_labels, predictions), 4))

# Printing a decision tree related to the trained model
plot_dtree(clf, "Tree")

# The accuracy (% of labels that match exactly) of this decision tree classifier is about 96.94% (rounded to 4 decimal places)
# This is really high number, meaning that in 96.94%
# of cases numbers 7 and 9 has been classified correctly (labels and predicions were the same)



# Create multinomial NB

# Instantiated a Naive Bayes calssifier and assigning it to variable gnb
gnb = MultinomialNB()

# Training model based on standarised training data and matching training labels
fitting_1 = gnb.fit(binary_class_training_data,binary_class_training_labels)

# Testing model with normalised testig data and appropriate testing labels
predictions_1 = gnb.predict(binary_class_testing_data)

# Computing and printing accuracy score
print("The accuracy of Naive Bayes classifier is", round(metrics.accuracy_score(binary_class_testing_labels, predictions_1),4))

# The accuracy of Naive Bayes classifier (about 91.59%) is lower than Decision Tree classifier by about 5% points.
# This difference is high, thus in case differentiating between numbers 7 and 9
# decision tree classifier would be preffered.
# There is no noticable difference in training times.



# Create a model with default parameters. Remember to set random state to 1234

# Instantiated the logistic regression function with given random state. Assigned instantiated function to variable lr
lr = LogisticRegression(random_state=(1234))

# Training the logistic regression classifier with normalised training data set and related labels
fitting_2 = lr.fit(binary_class_training_data,binary_class_training_labels)

# Testing fitted classifier with normalised testing data and labels
predictions_2 = lr.predict(binary_class_testing_data)

# Printing the accuracy of logistic regression classifier
print("The accuracy of Logistic Regression classifier is", round(metrics.accuracy_score(binary_class_testing_labels,predictions_2),4))

# The accuracy of Logistic Regression classifier higher than Naive Bayes classifier.
# The accuracy is still lower (by about 1% point) than the Decision Tree classifier.
# There is no noticable difference in training times.


# Create a random forest classifier with Default parameters

# Instantiated Random Forest classifier and assigning it to rf variable
rf = RandomForestClassifier()

# Training Random Forest classifier based on normalised training data and related labels
fitting = rf.fit(binary_class_training_data,binary_class_training_labels)

# Predicting labels of normalised testing data
predictions_3 = rf.predict(binary_class_testing_data)

# Printing the accuracy of the Random Forest classifier
print("The accuracy of Random Forest classifier is", round(metrics.accuracy_score(binary_class_testing_labels,predictions_3),4))

# The accuracy of random forest classifier is the highest among all implemented classifiers.
# The accuracy is higher by about 0.6% point than decision tree classifier (this difference may not be statistically significant).
# In general this classifier would be preffered to distinguish between numbers 7 and 9 within MINST dataset, due to high accuracy.
# There is no noticable difference in training times.

# Using previously obtained predictions, I display classification reports and confucion matix of all 4 classifiers

print("Classification report and confusion matrix for Decision Tree")
# Clafficitation report for Decision Tree
print(metrics.classification_report(binary_class_testing_labels, predictions))
# Confusion matrix for Decision Tree
print(metrics.confusion_matrix(binary_class_testing_labels, predictions))

print("Classification report and confusion matrix for Naive Bayes classifier ")
# Clafficitation report for Naive Bayes
print(metrics.classification_report(binary_class_testing_labels, predictions_1))
# Confusion matrix for Naive Bayes
print(metrics.confusion_matrix(binary_class_testing_labels, predictions_1))

print("Classification report and confusion matrix for Logistic Regression classifier ")
# Clafficitation report for Logistic Regression
print(metrics.classification_report(binary_class_testing_labels, predictions_2))
# Confusion matrix for Logistic Regression
print(metrics.confusion_matrix(binary_class_testing_labels, predictions_2))

print("Classification report and confusion matrix for Ransom Forest classifier ")
# Clafficitation report for Random Forest
print(metrics.classification_report(binary_class_testing_labels, predictions_3))
# Confusion matrix for Random Forest
print(metrics.confusion_matrix(binary_class_testing_labels, predictions_3))

# Classification report includes respectivelly:
# Precision: prpoportion of correct positive identifications
# recall: true positives among true positives and false negatives
# F1-score: harmonic mean of presicion and recall
# Support: number of samples of the true response that lie in a class

# Confusion matrix shows respectively: true positives (7), false positives (7 that are classified as 9)
# false negatives (9 that are classified as 7), true negatives (9)

# Both reports and confusion matricies show that a Random Forest classifier
# performs best, having the highest precision, recall and F1-score.



# Predicting probability for both 9 and 7 for all used classifiers
predictions=clf.predict_proba(binary_class_testing_data)
predictions_1=gnb.predict_proba(binary_class_testing_data)
predictions_2=lr.predict_proba(binary_class_testing_data)
predictions_3=rf.predict_proba(binary_class_testing_data)

# Choosing probabilities for 7
predictions=predictions[:,0]
predictions_1=predictions_1[:,0]
predictions_2=predictions_2[:,0]
predictions_3=predictions_3[:,0]

# Creating a plot with a given size
fig = plt.figure(figsize=(17,5))

# Setting a title for the set of plots
fig.suptitle('ROC Curves for Models 1 to 4 Respectively from the Left')

# Computing roc_curve components for model 1
fpr, tpr, thresholds = metrics.roc_curve(binary_class_testing_labels, predictions, pos_label=7)
# Plot 1 for Decision Tree Classifier
ax1 = plt.subplot(1, 4, 1)
ax1.set_xlabel('false positive rate') # axis titles
ax1.set_ylabel('true positive rate') # axis titles
ax1.plot(fpr, tpr) # plotting

# Computing roc_curve components for model 2
fpr_1, tpr_1, thresholds_1 = metrics.roc_curve(binary_class_testing_labels, predictions_1, pos_label=7)
ax1 = plt.subplot(1, 4, 2) # Plot 2 for Naive Bayes Classifier
ax1.set_xlabel('false positive rate') # axis titles
ax1.set_ylabel('true positive rate') # axis titles
ax1.plot(fpr_1, tpr_1) #plotting

# Computing roc_curve components for model 3
fpr_2, tpr_2, thresholds_2 = metrics.roc_curve(binary_class_testing_labels, predictions_2, pos_label=7)
ax2 = plt.subplot(1, 4, 3) # Plot 3 for Logistic Regression
ax2.set_xlabel('false positive rate') # axis titles
ax2.set_ylabel('true positive rate') # axis titles
ax2.plot(fpr_2, tpr_2) # plotting

# Computing roc_curve for model 4
fpr_3, tpr_3, thresholds_3 = metrics.roc_curve(binary_class_testing_labels, predictions_3, pos_label=7)
ax3= plt.subplot(1, 4, 4) # Plot 4 for Random Forest
ax3.set_xlabel('false positive rate') # axis titles
ax3.set_ylabel('true positive rate') # axis titles
ax3.plot(fpr_3, tpr_3) # plotting

plt.show() # Showing roc cuves for all models

# The most desirable model has roc curve with points closest to the left top of a plot.
# We intend to minimise the false-positive rate (x-axis) and maximise true-positive rate (y-axis).
# We can see that model 1 (Decision Tree), and model 5 (Random Forest) perform bests based on ROC curves.
# However, all curves are very similar due to good quality data set.
# Predicting probability for both 9 and 7 for all used classifiers
predictions_=clf.predict_proba(binary_class_testing_data)
predictions_1_=gnb.predict_proba(binary_class_testing_data)
predictions_2_=lr.predict_proba(binary_class_testing_data)
predictions_3_=rf.predict_proba(binary_class_testing_data)

# Choosing probabilities for 9
predictions_=predictions_[:,1]
predictions_1_=predictions_1_[:,1]
predictions_2_=predictions_2_[:,1]
predictions_3_=predictions_3_[:,1]

# Print AUC score of model 1 (Decision Tree)
print("Area under curve for Decision Tree is", round(metrics.roc_auc_score(binary_class_testing_labels, predictions_),4))
# Print AUC score of model 2 (Naive Bayes)
print("Area under curve for Naive Bayes is",round(metrics.roc_auc_score(binary_class_testing_labels, predictions_1_),4))
# Print AUC score of model 3 (Logistic Regression)
print("Area under curve for Logistic Regression is",round(metrics.roc_auc_score(binary_class_testing_labels, predictions_2_),4))
# Print AUC score of model 4 (Random Forest)
print("Area under curve for Random FOrest is", round(metrics.roc_auc_score(binary_class_testing_labels, predictions_3_),4))

# Area under the ROC curve ranges from 0 to 1. Area of 1 represents the perfect fit. Area 0.5 represents very poor fit (guess)
# Area under the ROC curve can be interpreted as the average value of sensitivity for all possible values of specificity
# We want to choose classifier which area is as close to 1 as possible
# Based on ROC curve, we should choose Random Forest classifier, as it has almost ideall fit - close to 1


from sklearn.utils.fixes import signature

# Obtaining data necesarry for recall curve of models 1 to 4 respectively

precision, recall, _ = metrics.precision_recall_curve(binary_class_testing_labels, predictions, pos_label=7)
precision_1, recall_1, _ = metrics.precision_recall_curve(binary_class_testing_labels, predictions_1, pos_label=7)
precision_2, recall_2, _ = metrics.precision_recall_curve(binary_class_testing_labels, predictions_2, pos_label=7)
precision_3, recall_3, _ = metrics.precision_recall_curve(binary_class_testing_labels, predictions_3, pos_label=7)

# Plotting recall curves based on above recall_cirve data

fig = plt.figure(figsize=(17,5))
fig.suptitle('Precision-Recall Curve for Models 1 to 4 Respectively from the Left Side')

step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})

# Plotting recall curve for Decision Tree
plt0 = plt.subplot(1, 4, 1)
plt0.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt0.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

plt0.set_xlabel('Recall')
plt0.set_ylabel('Precision')

# Plotting recall curve for Naive Bayes
plt1 = plt.subplot(1, 4, 2)
plt1.step(recall_1, precision_1, color='b', alpha=0.2,
         where='post')
plt1.fill_between(recall_1, precision_1, alpha=0.2, color='b', **step_kwargs)

plt1.set_xlabel('Recall')
plt1.set_ylabel('Precision')

# Plotting recall curve for Logistic Regression
plt2 = plt.subplot(1, 4, 3)
plt2.step(recall_2, precision_2, color='b', alpha=0.2,
         where='post')
plt2.fill_between(recall_2, precision_2, alpha=0.2, color='b', **step_kwargs)

plt2.set_xlabel('Recall')
plt2.set_ylabel('Precision')

# Plotting recall curve for Ransom Forest
plt3 = plt.subplot(1, 4, 4)
plt3.step(recall_3, precision_3, color='b', alpha=0.2,
         where='post')
plt3.fill_between(recall_3, precision_3, alpha=0.2, color='b', **step_kwargs)

plt3.set_xlabel('Recall')
plt3.set_ylabel('Precision')

plt.show() # Display the plotted figures

# High precision means low false positive rate
# High recall means low false negative rate
# The most derirable estimator have both high precision and high recall, may be measured by area under the curve
# Based on above theory, we may see that model 1 (decision tree) and model 4 (random forest) perform best

# Tuning Random Forest for MNIST
tuned_parameters = [{'max_features': ['sqrt', 'log2'], 'n_estimators': [1000, 1500]}]


# Initialise the grid search with parameters: random forest classifier,
# cv = 3, verbose = 3 and param_grid = tuden_parameters
clf = GridSearchCV(estimator = RandomForestClassifier(), param_grid = tuned_parameters, cv = 3, verbose = 2)
# Fitting all data (normalised) to the model
clf.fit(all_scaled_data, all_scaled_target)
# We dont need testing data, because cross validation divides the whole dataset into smaller parts
# and then those smaller parts are divided into testing and training data

# print the details of the best model and its accuracy

# Details of the best chosen estimator
print("The parameters of best chosen estimator are: ",clf.best_estimator_)
# Print accuracy of best model
print("The score of best model is ", round(clf.best_score_),4)


# In case of grid search, we reach the accuracy score of 0.99 for the best model
# (which specificaiton is printed below)
# This means that we have predicted almost all of the data correctly
# However, the training time is significantly longer, as we test many models
# This model gives the best overall accuracy
# If training time is not a problem, or we have enough computational power, cross-validation
# would be preffered method, as it gives best accuracy result.

```

``` python
%matplotlib inline

#Array processing
import numpy as np

#Data analysis, wrangling and common exploratory operations
import pandas as pd
from pandas import Series, DataFrame

#For visualization. Matplotlib for basic viz and seaborn for more stylish figures + statistical figures not in MPL.
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.core.display import Image

from sklearn.datasets import fetch_mldata
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.svm import SVC, LinearSVC , SVR
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.model_selection import KFold, train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import pydot, io
import time

#######################End imports###################################



# Instantiate OneVsRestClassifier with argument LinearSVC()
ovr = OneVsRestClassifier(LinearSVC())
# Start measuring time of execution of below command
start = time.time()
# Train the multi-class classifier on normalised training data
ovr.fit(train_data, train_labels)
# Stop measuring time of execution of above command
end = time.time()
# Using above model to predict labels of normalised test data
predict = ovr.predict(test_data)
# Printing required information
print("The accuracy of OneVsRest classifier is", round(metrics.accuracy_score(predict, test_labels),4), "and training time is", round(end - start,2), "seconds")

# The accurace of a multi-class classifier is about 91.15% and training time is about 113 seconds.
# It obviously take longer to train it compared to the case where we just needed to differenciate between 7 and 9
# because now we consider the whole dataset now (and have multiple classess).


# Reducing the dimensionality using PCA, limited dimensionality to 100 features
pca = PCA(n_components=100)
# Determine most important features of train data and reduce dimensionality of training data set
ptd = pca.fit_transform(train_data)
# Reduce dimentionality of test_data
pdd = pca.transform(test_data)
# Start measuring time of below operation
start = time.time()
# Training the multi-class classifier defined in previous question (on dataset with reduced dimensions)
ovr.fit(ptd, train_labels)
# Finish measuring time
end = time.time()
# Predict labels of transformed testing data (with 100 fetures). Based on model defined in previous question
predict = ovr.predict(pdd)

# Print classification accuracy and training time
print("The accuracy of OneVsRest classifier with data having 100 features is ", round(metrics.accuracy_score(predict, test_labels),4), "and training time is ", round(end - start,2), "seconds")

# We may note that because of reduced dimensionality of data, the training time has decreassed by more than 20 seconds.
# Accuracy remains very similar (91.15% vs 90.74%).
# Dimensionality reduction from 784 features to 100 features lead to
# improvement of training time without significant loss of accuracy.
# This means that PCA improves the efficiency of classification in relation
# to training time and amount of data needed



print("The variance explained by first 100 principal components is", round(sum(pca.explained_variance_ratio_),4))

# The culminative variance explained by the first 100 principal components is about 91.47%.
# This value is lower than 95% treshold.

# Ideally we would want to increase the number of principal components
# so that its n components explain 95% of variance.

# This could be done by specyfying the argument inside the PCA function as: PCA(n_components=0.99, svd_solver='full').
# This, however, could increase significanly the amount of data neded (and training time)
# just to gain small increase in variance explained.

# Because of it, I would recommend keeping just 100 features, as it is enough to
# make sensible classification with high accuracy.


```
