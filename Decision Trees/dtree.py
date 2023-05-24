import numpy as np
from scipy import stats
from sklearn.metrics import r2_score, accuracy_score

class DecisionNode:
    def __init__(self, col, split, lchild, rchild):
        self.col = col
        self.split = split
        self.lchild = lchild
        self.rchild = rchild

    def predict(self, x_test):
        # Make decision based upon x_test[col] and split
        #if type(self) == LeafNode()
        predictions = []
        
        for i in range(0,x_test.shape[0]): 
            tree = self
            while type(tree)!=LeafNode:
                col = tree.col
                split = tree.split
                lchild = tree.lchild
                rchild = tree.rchild
                if x_test[i,col]<split:
                    tree = lchild
                else:
                    tree = rchild
            predictions.append(tree.predicted_values)
        return predictions
            


class LeafNode:
    def __init__(self, y, prediction):
        "Create leaf node from y values and prediction; prediction is mean(y) or mode(y)"
        self.n = len(y)
        self.prediction = prediction

    def predict(self, x_test):
        # return prediction
        ...
        if self.prediction == np.var:
            self.predicted_values = x_test.mean()
        else:
            self.predicted_values = stats.mode(x_test,keepdims = True)[0].sum()
            
        


def gini(x):
    """
    Return the gini impurity score for values in y
    Assume y = {0,1}
    Gini = 1 - sum_i p_i^2 where p_i is the proportion of class i in y
    """
    unique, counts = np.unique(x, return_counts=True)
    return 1 - np.square(counts/counts.sum()).sum()
    ...

    
def find_best_split(X, y, loss, min_samples_leaf):
    ...
    features = X.shape[1]
    best = (-1,-1,loss(y))
    if loss(y) == 0:
        return -1,-1
    for feature in range(0,features):
        candidates_split = np.unique(np.random.choice(np.unique(X[:,feature]),11))
       # candidates_split = np.unique(X[:,feature])
        for split in candidates_split:
            yl = y[X[:,feature]<split]
            yr = y[X[:,feature]>=split]
            if len(yl)<min_samples_leaf and len(yr)<min_samples_leaf:
                continue
            loss_feature = (loss(yl)*len(yl) + loss(yr)*len(yr))/(len(yl) + len(yr))
            if loss_feature == 0:
                return feature,split
            if loss_feature < best[2]:
                best = (feature,split,loss_feature)
    
    return best[0],best[1]
                
            
    
    
class DecisionTree621:
    def __init__(self, min_samples_leaf=1, loss=None):
        self.min_samples_leaf = min_samples_leaf
        self.loss = loss # loss function; either np.var for regression or gini for classification
        
    def fit(self, X, y):
        """
        Create a decision tree fit to (X,y) and save as self.root, the root of
        our decision tree, for  either a classifier or regression.  Leaf nodes for classifiers
        predict the most common class (the mode) and regressions predict the average y
        for observations in that leaf.

        This function is a wrapper around fit_() that just stores the tree in self.root.
        """
        self.root = self.fit_(X, y)


    def fit_(self, X, y):
        """
        Recursively create and return a decision tree fit to (X,y) for
        either a classification or regression.  This function should call self.create_leaf(X,y)
        to create the appropriate leaf node, which will invoke either
        RegressionTree621.create_leaf() or ClassifierTree621.create_leaf() depending
        on the type of self.

        This function is not part of the class "interface" and is for internal use, but it
        embodies the decision tree fitting algorithm.

        (Make sure to call fit_() not fit() recursively.)
        """
        if X.shape[0]<self.min_samples_leaf or len(np.unique(X)) == 1:
            leaf = LeafNode(y,self.loss)
            leaf.predict(y)
            return leaf
        col,split = find_best_split(X, y, self.loss, self.min_samples_leaf)
        if col == -1:
            leaf = LeafNode(y,self.loss)
            leaf.predict(y)
            return leaf
        lchild = self.fit_(X[X[:,col] < split],y[X[:,col] < split])
        rchild = self.fit_(X[X[:,col] >= split],y[X[:,col] >= split])
        return DecisionNode(col,split,lchild,rchild)

    def predict(self, X_test):
        """
        Make a prediction for each record in X_test and return as array.
        This method is inherited by RegressionTree621 and ClassifierTree621 and
        works for both without modification!
        """
        predictions = []
        for i in range(0,X_test.shape[0]): 
            tree = self.root
            while type(tree)!=LeafNode:
                col = tree.col
                split = tree.split
                lchild = tree.lchild
                rchild = tree.rchild
                if X_test[i,col]<split:
                    tree = lchild
                else:
                    tree = rchild
            predictions.append(tree.predicted_values)
        return predictions


class RegressionTree621(DecisionTree621):
    def __init__(self, min_samples_leaf=1):
        super().__init__(min_samples_leaf, loss=np.var)

    def score(self, X_test, y_test):
        "Return the R^2 of y_test vs predictions for each record in X_test"
        
        return r2_score(self.predict(X_test),y_test)

    def create_leaf(self, y):
        """
        Return a new LeafNode for regression, passing y and mean(y) to
        the LeafNode constructor.
        """
        return LeafNode(y,self.loss)

class ClassifierTree621(DecisionTree621):
    def __init__(self, min_samples_leaf=1):
        super().__init__(min_samples_leaf, loss=gini)

    def score(self, X_test, y_test):
        "Return the accuracy_score() of y_test vs predictions for each record in X_test"
        ...
        
        return accuracy_score(self.predict(X_test),y_test)

    def create_leaf(self, y):
        """
        Return a new LeafNode for classification, passing y and mode(y) to
        the LeafNode constructor. Feel free to use scipy.stats to use the mode function.
        """
        return LeafNode(y,self.loss)
