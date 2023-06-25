import numpy as np
from sklearn.tree import DecisionTreeClassifier
from pathlib import Path

def accuracy(y, pred):
    return np.sum(y == pred) / float(len(y))

def parse_spambase_data(filename):
    """ Given a filename return X and Y numpy arrays

    X is of size number of rows x num_features
    Y is an array of size the number of rows
    Y is the last element of each row. (Convert 0 to -1)
    """
    X = np.loadtxt(filename, delimiter=',')
    X, Y = X[:, :-1], (-1)**(X[:, -1]-1)
    return X, Y


def adaboost(X, y, num_iter, max_depth=1):
    """Given an numpy matrix X, a array y and num_iter return trees and weights 
   
    Input: X, y, num_iter
    Outputs: array of trees from DecisionTreeClassifier
             trees_weights array of floats
    Assumes y is {-1, 1}
    """
    trees = []
    trees_weights = [] # alphas
    N, _ = X.shape
    d = np.ones(N) / N
    
    for i in range(num_iter):
        tree = DecisionTreeClassifier(max_depth=max_depth)
        tree.fit(X, y, sample_weight=d)
        trees.append(tree)
        missed = np.where(accuracy(tree.predict(X), y))[0]
        err = d[missed] / (np.sum(d)+0.00001)
        alpha = np.log((1-err)/err)
        trees_weights.append(alpha)
        d[missed] *= np.log(alpha)
    return trees, trees_weights


def adaboost_predict(X, trees, trees_weights):
    """Given X, trees and weights predict Y
    """
    # X input, y output
    N, _ =  X.shape
    y = np.zeros(N)
    for t in range(len(trees)):
        y += (trees[t].predict(X)*trees_weights[t])
    y /= np.abs(y)
    return y
