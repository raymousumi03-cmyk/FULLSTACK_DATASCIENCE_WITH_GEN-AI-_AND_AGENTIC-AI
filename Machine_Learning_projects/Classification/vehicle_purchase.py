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


# train the model x_train and y_train

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()
classifier.fit(x_train,y_train)


# Create prediction for x_test

y_pred = classifier.predict(x_test)


# Making the confusion matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)


# Model accurecy

from sklearn.metrics import accuracy_score

ac = accuracy_score(y_test,y_pred)
print(ac)


# classification report

from sklearn.metrics import classification_report

cr = classification_report(y_test, y_pred)
print(cr)



bias = classifier.score(x_train,y_train)
bias


variance = classifier.score(x_test,y_test)
variance




#------------------------FUTURE PrEDICTION--------------------------------------------------------------

ds1 = pd.read_csv(r"C:\Users\raymo\NareshIT_DataScience_AI\Machine_Learning_Projects\Classification\final1.csv")
ds2= ds1.copy()
ds1 = ds1.iloc[:,[3,4]].values


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()


m = sc.fit_transform(ds1)

y_pred1 = pd.DataFrame()


ds2 ['y_pred1'] =classifier.predict(m)

ds2.to_csv('final.csv')

import os 
os.getcwd()
