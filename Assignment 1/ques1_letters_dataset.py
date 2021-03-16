# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 02:27:01 2021

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

print("Implementation of k-Times Markov Sampling for SVMC")

letters = pd.read_csv(r"C:\Users\user\Downloads\letter-recognition.csv")


letters.columns = ['lettr', 'x-box', 'y-box', 'width', 'high', 'onpix', 'x-bar',
       'y-bar', 'x2bar', 'y2bar', 'xybar', 'x2ybar', 'xy2bar', 'x-ege',
       'xegvy', 'y-ege', 'yegvx']
# print(letters.columns)

order = list(np.sort(letters['lettr'].unique()))

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

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2, random_state = 101)

print("Enter the value of k")
k = int(input())
i = 0
model_linear = SVC(kernel='linear')
while(i < k):
    model_linear.fit(X_train, y_train)
    y_pred = model_linear.predict(X_test)
    i+=1
    
# print()
print("%accuracy with linear kernel:", metrics.accuracy_score(y_true=y_test, y_pred=y_pred), "\n")

i = 0
non_linear_model = SVC(kernel='rbf')
while(i < k):
    non_linear_model.fit(X_train, y_train)    
    y_pred = non_linear_model.predict(X_test)
    i+=1
    
print("%accuracy with RBF kernel:", metrics.accuracy_score(y_true=y_test, y_pred=y_pred), "\n")

i = 0
non_linear_model1 = SVC(kernel='poly', degree = 2)
while(i < k):
    non_linear_model1.fit(X_train, y_train)
    y_pred = non_linear_model1.predict(X_test)
    i+=1

print("%accuracy with Polynomial kernel (x^2):", metrics.accuracy_score(y_true=y_test, y_pred=y_pred), "\n")

















































