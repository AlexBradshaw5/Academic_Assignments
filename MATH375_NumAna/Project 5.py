# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 15:23:57 2021

@author: alexb

Linear Regression
"""

import numpy as np
import sklearn as sk
import sklearn.datasets
import sklearn.model_selection
import matplotlib.pyplot as plt

Question = 3

if(Question == 3):
    dataset = sk.datasets.fetch_openml('MNIST_784', as_frame = False)
if(Question == 2):
    dataset = sk.datasets.load_breast_cancer()


X = dataset.data
Y = dataset.target
if(Question == 3):
    labels = dataset.target
    N = len(labels)
    Y = np.zeros(N)
    for i in range(N):
        if(labels[i] != '0'):
            Y[i] = 1
    X = X/255
    
X_train,X_val,Y_train,Y_val = \
    sk.model_selection.train_test_split(X,Y,train_size=0.7, random_state = 123)

'''
img = X[1]
plt.imshow(img.reshape(28,28), cmap = 'gray')

Different way to split

N_train = 5000
X_train = X[0:N_train]
X_val = X[N_train:]

'''

'''
#standardize
mu = np.mean(X_train,axis=0)
s = np.std(X_train,axis=0)
X_train = (X_train-mu)/s
X_val = (X_val - mu)/s
'''

X_train = np.insert(X_train,0,1,axis=1)
X_val = np.insert(X_val,0,1,axis=1)

def eval_L(beta, x, y):
    N = x.shape[0]
    total = 0
    for i in range(N):
        total += cross_entropy(y[i],sigmoid((x[i].T)@beta))
    return total/N
   
def grad_L(beta,x,y):
    N = x.shape[0]
    total = 0
    for i in range(N):
        total += x[i]*(sigmoid(x[i].T@beta)-y[i])
    return total/N

def cross_entropy(p,q):
    return -p*np.log(q) - (1-p)*np.log(1-q)
        
def sigmoid(u):
    eu = np.exp(u)
    return eu / (1+eu)

betak = np.zeros(X.shape[1]+1)
max_iter = 100
t = 1e-3
LVals = []

#gradient descent
for i in range(max_iter):
    betak = betak - t*grad_L(betak, X_train, Y_train)
    LVals.append(eval_L(betak, X_train,Y_train))
    print("Iteration: ",i+1)

y_test = sigmoid(X_val @ betak)
correct = 0.
total = 0.
if(Question == 2):
    for i in range(len(y_test)):
        yi = round(y_test[i])
        if(yi == Y_val[i]):
            correct += 1
        total += 1
   
if(Question == 3):
    for i in range(len(y_test)):
        yhat = round(y_test[i])
        yi = Y_val[i]
        if(yi == yhat):
            correct +=1
        total += 1
        
        
plt.plot(LVals)    
print(correct/total * 100, "% Accuracy")
plt.semilogy()

'''
Accuracy for:
0 or not - 91.255%
1 or not - 88.767%
2 or not - 90.152%
3 or not - 89.705%
4 or not - 90.243%
5 or not - 90.962%
6 or not - 90.229%
7 or not - 89.619%
8 or not - 90.219%
9 or not - 89.890%

The most difficult classifications for my model to make were 1,7,3,9, and 6.
1 and 7 make sense because they look similar in that they have one stroke down and sometimes a small hat
3 looks like a 9 missing half of it's hat and 6 kinda looks like 5 but also an upside down 9. 

''' 
        