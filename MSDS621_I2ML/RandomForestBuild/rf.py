import numpy as np
from sklearn.utils import resample

from dtree import *

class RandomForest621:
    def __init__(self, n_estimators=10, oob_score=False):
        self.n_estimators = n_estimators
        self.oob_score = oob_score
        self.oob_score_ = np.nan

    def fit(self, X, y):
        """
        Given an (X, y) training set, fit all n_estimators trees to different,
        bootstrapped versions of the training data. Keep track of the indexes of
        the OOB records for each tree.  After fitting all of the trees in the forest,
        compute the OOB validation score estimate and store as self.oob_score_, to
        mimic sklearn.
        """
        for t in range(self.n_estimators):
            samples = np.random.choice(np.arange(X.shape[0]), size = X.shape[0])
            self.trees[t].fit(X[samples], y[samples])
            self.trees[t].oobs = np.setdiff1d(np.arange(X.shape[0]), samples)
        if self.oob_score:
            self.oob_score_ = self.compute_oob(X, y)
            
            
            
            
class RandomForestRegressor621(RandomForest621):
    def __init__(self, n_estimators=10, min_samples_leaf=3, 
    max_features=0.3, oob_score=False):
        super().__init__(n_estimators, oob_score=oob_score)
        self.trees = [RegressionTree621(min_samples_leaf=min_samples_leaf, max_features=max_features)\
                      for i in range(self.n_estimators)]

    def predict(self, X_test) -> np.ndarray:
        """
        Given a 2D nxp array with one or more records, compute the weighted average
        prediction from all trees in this forest. Weight each trees prediction by
        the number of observations in the leaf making that prediction.  Return a 1D vector
        with the predictions for each input record of X_test.
        """
        preds = [[] for i in range(len(X_test))]
        weights = [[] for i in range(len(X_test))]
        for i in range(len(X_test)):
            leaves = [t.leaf(X_test[i]) for t in self.trees] # 10 leaves
            [[preds[i].append(leaf.prediction), weights[i].append(leaf.n)] for leaf in leaves]
        return np.array([(np.dot(preds[i], weights[i])/np.sum(weights[i])) for i in range(len(preds))])
            
            
    def score(self, X_test, y_test) -> float:
        """
        Given a 2D nxp X_test array and 1D nx1 y_test array with one or more records,
        collect the prediction for each record and then compute R^2 on that and y_test.
        """
        yhats = self.predict(X_test)
        return r2_score(y_test, yhats)
    
    def compute_oob(self, X, y):
        preds, counts = np.zeros(X.shape[0]), np.zeros(X.shape[0])
        for tree in self.trees:
            preds[tree.oobs]+=tree.predict(X[tree.oobs])
            counts[tree.oobs]+=1
        keepers = np.where(counts>0)[0]
        final = preds[keepers]/counts[keepers]
        return r2_score(y[keepers], final)
    
class RandomForestClassifier621(RandomForest621):
    def __init__(self, n_estimators=10, min_samples_leaf=3, 
    max_features=0.3, oob_score=False):
        super().__init__(n_estimators, oob_score=oob_score)
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.trees = [ClassifierTree621(min_samples_leaf=min_samples_leaf, max_features=max_features)\
                      for i in range(self.n_estimators)]

    def predict(self, X_test) -> np.ndarray:
        preds = [0 for i in range(len(X_test))]
        for i in range(len(X_test)):
            class_counts = np.zeros(20)
            leaves = [t.leaf(X_test[i]) for t in self.trees] # 10 leaves
            total_list = []
            [total_list.extend(leaves[i].y) for i in range(len(leaves))]
            value_counts = np.unique(total_list, return_counts=True)
            class_counts[value_counts[0]]+=value_counts[1]
            preds[i] = class_counts.argmax()
        return preds
    
    def score(self, X_test, y_test) -> float:
        """
        Given a 2D nxp X_test array and 1D nx1 y_test array with one or more records,
        collect the predicted class for each record and then compute accuracy between
        that and y_test.
        """
        yhats = self.predict(X_test)
        return accuracy_score(y_test, yhats)
    
    def compute_oob(self, X, y):
        preds, counts = [[] for i in range(X.shape[0])], np.zeros(X.shape[0])
        for tree in self.trees:
            yhats = tree.predict(X[tree.oobs])
            [preds[tree.oobs[i]].extend([yhats[i]]) for i in range(len(tree.oobs))]
            counts[tree.oobs]+=1
        keepers = np.where(counts>0)[0]
        final = [stats.mode(preds[i])[0][0] for i in keepers]
        return accuracy_score(y[keepers], final)

