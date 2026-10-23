import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv(r"C:\Users\raymo\NareshIT_DataScience_AI\Machine_Learning_Projects\Classification\xgboost\Churn_Modelling.csv")


x = dataset.iloc[:, 3:-1].values 
y = dataset.iloc[:, -1].values


print(x)
print(y)

# Encoding the categorical column
# label encoding the "gender" column

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x[:,2] = le.fit_transform(x[:,2])


# one hot encoding in the "geography" column

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[1])],remainder='passthrough')
x= np.array(ct.fit_transform(x))



from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,
                                               test_size=0.2,
                                               train_size=0.8,
                                               random_state=0) 


# Training Xgboost on the training set

from xgboost import XGBClassifier

classifier = XGBClassifier(n_estimator=22)
classifier.fit(x_train,y_train)


# Predicting the test set results

y_pred = classifier.predict(x_test)


# Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)


# Model accurecy

from sklearn.metrics import accuracy_score

ac = accuracy_score(y_test,y_pred)
print(ac)


bias = classifier.score(x_train,y_train)
bias


variance = classifier.score(x_test,y_test)
variance



from sklearn.metrics import roc_auc_score, roc_curve

y_pred = classifier.predict_proba(x_test)[:,1]

auc_score = roc_auc_score(y_test,y_pred)
print(auc_score)


fpr, tpr , thresholds = roc_curve(y_test,y_pred)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {auc_score:.2f})')
plt.plot([0,1], [0,1], 'k--')  # Random classifier line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid()
plt.show() 