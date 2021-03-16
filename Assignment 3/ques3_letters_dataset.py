# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 21:53:38 2021

@author: user
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import scale


print("Implementation of Markov Sampling for SVMC")

# dataset
letters = pd.read_csv(r"C:\Users\user\Downloads\letter-recognition.csv")



# print("Dimensions: ", letters.shape, "\n")

# data types
# print(letters.info())

# head
# letters.head()
# print(letters.columns)

letters.columns = ['lettr', 'x-box', 'y-box', 'width', 'high', 'onpix', 'x-bar',
       'y-bar', 'x2bar', 'y2bar', 'xybar', 'x2ybar', 'xy2bar', 'x-ege',
       'xegvy', 'y-ege', 'yegvx']
# print(letters.columns)

order = list(np.sort(letters['lettr'].unique()))
# print(order)

plt.figure(figsize=(16, 8))
sns.barplot(x='lettr', y='x-box', 
            data=letters, 
            order=order)



letter_means = letters.groupby('lettr').mean()
letter_means.head()

plt.figure(figsize=(18, 10))
sns.heatmap(letter_means)

round(letters.drop('lettr', axis=1).mean(), 2)

X = letters.drop("lettr", axis = 1)
y = letters['lettr']

X_scaled = scale(X)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.3, random_state = 101)

model_linear = SVC(kernel='linear')
model_linear.fit(X_train, y_train)

# predict
y_pred = model_linear.predict(X_test)
print()
print("%accuracy of linear kernel:", metrics.accuracy_score(y_true=y_test, y_pred=y_pred), "\n")

# cm
# print(metrics.confusion_matrix(y_true=y_test, y_pred=y_pred))

non_linear_model = SVC(kernel='rbf')

# fit
non_linear_model.fit(X_train, y_train)

# predict
y_pred = non_linear_model.predict(X_test)

print("%accuracy with RBF kernel:", metrics.accuracy_score(y_true=y_test, y_pred=y_pred), "\n")

# cm
# print(metrics.confusion_matrix(y_true=y_test, y_pred=y_pred))

non_linear_model1 = SVC(kernel='poly', degree = 2)

# fit
non_linear_model1.fit(X_train, y_train)

y_pred = non_linear_model1.predict(X_test)
print("%accuracy of Polynomial kernel (x^2):", metrics.accuracy_score(y_true=y_test, y_pred=y_pred), "\n")

















































