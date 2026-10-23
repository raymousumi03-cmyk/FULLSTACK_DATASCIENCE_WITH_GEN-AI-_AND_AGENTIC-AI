import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv(r"C:\Users\raymo\NareshIT_DataScience_AI\Machine_Learning_Projects\Classification\logit classification.csv")

x = dataset.iloc[:, [2,3]].values 
y = dataset.iloc[:, -1].values


# 80-20 Train-test split

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,
                                               test_size=0.2,
                                               train_size=0.8,
                                               random_state=0) 


# Scale the data

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# KNN classifier algorithm

from sklearn.neighbors import KNeighborsClassifier
classifier_knn = KNeighborsClassifier()
classifier_knn.fit(x_train,y_train)


# Predicting the test set result

y_pred = classifier_knn.predict(x_test)



# Making the confusion matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)


# Model accurecy

from sklearn.metrics import accuracy_score

ac = accuracy_score(y_test,y_pred)
print(ac)


bias = classifier_knn.score(x_train,y_train)
bias


variance = classifier_knn.score(x_test,y_test)
variance
