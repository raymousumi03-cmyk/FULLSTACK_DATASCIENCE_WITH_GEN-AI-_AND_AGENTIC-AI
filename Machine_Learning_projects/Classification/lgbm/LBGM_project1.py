import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv(r"C:\Users\raymo\NareshIT_DataScience_AI\Machine_Learning_Projects\Classification\Lgbm\Breast_cancer_data.csv")


x = dataset.iloc[:, 0:5].values 
y = dataset.iloc[:, -1].values


print(x)
print(y)


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,
                                               test_size=0.2,
                                               train_size=0.8,
                                               random_state=0) 




# build the lightgbm model

import lightgbm as lgb

clf = lgb.LGBMClassifier()
clf.fit(x_train, y_train)


# predict the results

y_pred=clf.predict(x_test)


# Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)


# Model accurecy

from sklearn.metrics import accuracy_score

ac = accuracy_score(y_test,y_pred)
print(ac)


bias = clf.score(x_train,y_train)
bias


variance = clf.score(x_test,y_test)
variance

