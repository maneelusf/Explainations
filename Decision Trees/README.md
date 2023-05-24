## About
This repository contains the implementation of Decision Trees. To use this library, call the RegressionTree621(For DecisionTreeRegressor) and Classifier621(for DecisionTreeClassifier) from the 
dtree.py file.

## Example
```sh
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
dt = RegressionTree621()
dt.fit(X_train, y_train)
score = dt.score(X_test, y_test)
```
