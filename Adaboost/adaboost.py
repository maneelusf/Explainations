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
    with open(filename,'r') as f:
        file_name = f.read().split('\n')
        file_name = [x.split(',') for x in file_name]
        X = np.array([x[:57] for x in file_name if len(x)==58]).astype(float)
        Y = np.array([x[-1]  for x in file_name if len(x)==58]).astype(int)
        Y = np.where(Y == 0,-1,Y)
    ### BEGIN SOLUTION
    
    ### END SOLUTION
    return X, Y


def adaboost(X, y, num_iter, max_depth=1):
    """Given an numpy matrix X, a array y and num_iter return trees and weights 
   
    Input: X, y, num_iter
    Outputs: array of trees from DecisionTreeClassifier
             trees_weights array of floats
    Assumes y is {-1, 1}
    """
    N, _ = X.shape
    trees = []
    d = np.ones(N) / N
    alphas = []
    for iteration in range(0,num_iter):
        tree = DecisionTreeClassifier(max_depth = max_depth)
        tree.fit(X,y,sample_weight = d)
        y_pred = tree.predict(X)
        error = sum(d[y_pred != y])/sum(d)
        alpha = np.log((1 - error)/(error))
        if alpha == np.inf:
            trees = [tree]
            alphas = [1]
            break
        weight_multiplier = np.exp(np.array([alpha if y==False else 0 for y in (y_pred == y)]))
        d = np.multiply(d,weight_multiplier)
        trees.append(tree)
        alphas.append(alpha)
    ### BEGIN SOLUTION
    
    ### END SOLUTION
    return trees,alphas


def adaboost_predict(X, trees, trees_weights):
    """Given X, trees and weights predict Y
    """
    # X input, y output
    N, _ =  X.shape
    y = np.zeros(N)
    #import pdb;pdb.set_trace()
    pred = np.array([trees_weight*tree.predict(X) for tree,trees_weight in zip(trees,trees_weights)])
    pred = pred.sum(axis = 0)
    pred = np.where(pred>0,1,-1)
    ### BEGIN SOLUTION
    
    ### END SOLUTION
    return pred
