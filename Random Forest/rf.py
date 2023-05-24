import numpy as np
from sklearn.utils import resample

from dtree import *
from scipy import stats

class RandomForest621:
    def __init__(self,n_estimators=10, oob_score=False):
        self.n_estimators = n_estimators
        self.oob_score = oob_score
        self.oob_score_ = np.nan
    def fit(self, X, y):
        """
        Given an (X, y) training set, fit all n_estimators trees to different,
        bootstrapped versions of the training data.  Keep track of the indexes of
        the OOB records for each tree.  After fitting all of the trees in the forest,
        compute the OOB validation score estimate and store as self.oob_score_, to
        mimic sklearn.
        """
        ...
        self.tree = []
        indices = np.arange(X.shape[0])
        oob_tree_indices = []
        
        for tree_number in range(self.n_estimators):
            bootstrapped_X,bootstrapped_y,bootstrapped_indices = resample(X,y,indices,n_samples = X.shape[0])
            if type(self) == RandomForestRegressor621:
                current_tree = RegressionTree621(self.min_samples_leaf,self.max_features)
                current_tree.fit(bootstrapped_X,bootstrapped_y)
                self.tree.append(current_tree)
                if self.oob_score:
                    test_indices = np.setdiff1d(indices,bootstrapped_indices)
                    oob_tree_indices.append(test_indices)               
            else:
                current_tree = ClassifierTree621(self.min_samples_leaf,self.max_features)
                current_tree.fit(bootstrapped_X,bootstrapped_y)
                self.tree.append(current_tree)
                if self.oob_score:
                    test_indices = np.setdiff1d(indices,bootstrapped_indices)
                    oob_tree_indices.append(test_indices)
        
        final_pred = []
                    
        if self.oob_score:
            for obs in range(0,len(X)):
                obs_pred = []
                tree_list = [True if obs in x else False for x in oob_tree_indices]
                if len(tree_list)!=0:
                    oob_trees =np.argwhere(tree_list).T[0]
                    for current_tree in oob_trees:
                        pred = self.tree[current_tree].predict_obs(X[obs:obs+1])
                        if type(self) == RandomForestRegressor621:
                            obs_pred.append((pred.mean(),len(pred)))
                        else:
                            obs_pred.append(pred)
                if type(self) == RandomForestRegressor621:
                    total_weights = [x[1] for x in obs_pred]
                    total_predictions = [x[0]*x[1] for x in obs_pred]
                    if sum(total_weights) == 0:
                        final_pred.append(0)
                    else:
                        final_pred.append(sum(total_predictions)/sum(total_weights))
                else:
                    if len(obs_pred) == 0:
                        final_pred.append(0)
                    else:
                        total_predictions = np.concatenate(obs_pred,axis = 1)
                        final_pred.append(modecalculation(total_predictions))
            if type(self) == RandomForestRegressor621:
                self.oob_score_ = r2_score(y,final_pred)
            else:
                self.oob_score_ = accuracy_score(y,final_pred)
                    

            
            
class RandomForestRegressor621(RandomForest621):
    def __init__(self, n_estimators=10, min_samples_leaf=3, max_features=0.3, oob_score=False):
        self.min_samples_leaf = min_samples_leaf
        loss = np.var
        self.max_features=max_features

        super().__init__(n_estimators, oob_score=oob_score)

    def predict(self, X_test) -> np.ndarray:
        """
        Given a 2D nxp array with one or more records, compute the weighted average
        prediction from all trees in this forest. Weight each trees prediction by
        the number of observations in the leaf making that prediction.  Return a 1D vector
        with the predictions for each input record of X_test.
        """
        predictions = []
        weights = []
        for tree in self.tree:
            a = tree.predict(X_test)
            b = tree.predict_obs(X_test)
            predictions.append(a)
            weights.append([len(x) for x in b])
        predictions = np.array(predictions)
        weights = np.array(weights)
        
        return (predictions*weights).sum(axis = 0)/weights.sum(axis = 0)
        
        
        
    def score(self, X_test, y_test) -> float:
        """
        Given a 2D nxp X_test array and 1D nx1 y_test array with one or more records,
        collect the prediction for each record and then compute R^2 on that and y_test.
        """
        ...
        return r2_score(y_test,self.predict(X_test))
    
class RandomForestClassifier621(RandomForest621):
    def __init__(self, n_estimators=10, min_samples_leaf=3, max_features=0.3, oob_score=False):
        self.min_samples_leaf = min_samples_leaf
        loss = gini
        self.max_features=max_features

        super().__init__(n_estimators, oob_score=oob_score)

    def predict(self, X_test) -> np.ndarray:
        """
        Given a 2D nxp array with one or more records, compute the weighted average
        prediction from all trees in this forest. Weight each trees prediction by
        the number of observations in the leaf making that prediction.  Return a 1D vector
        with the predictions for each input record of X_test.
        """
        predictions = []
        for tree in self.tree:
            predictions.append(tree.predict_obs(X_test))
        final_pred = []
        for obs in range(0,len(predictions[0])):
            obs_list = [predictions[tree][obs] for tree in range(0,len(self.tree))]
            final_pred.append(np.concatenate(obs_list))

        final_pred = [modecalculation(obs) for obs in final_pred]
            
        
        return np.array(final_pred)
    
    def score(self, X_test, y_test) -> float:
        """
        Given a 2D nxp X_test array and 1D nx1 y_test array with one or more records,
        collect the prediction for each record and then compute R^2 on that and y_test.
        """
        ...
        return accuracy_score(y_test,self.predict(X_test))
        
def modecalculation(a):
    unique = np.unique(a,return_counts = True)
    return unique[0][np.argmax(unique[1])]
    
        
