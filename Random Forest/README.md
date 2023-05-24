## About
This repository contains the implementation of Random Forests. To use this library, call the RandomForestRegressor621(For RandomForestRegressor) and RandomForestClassifier621(for RandomForestClassifier) from the 
rf.py file. Additional hyperparameters supported are 
- Minimum samples per leaf(min_samples_leaf)
- Number of trees(n_estimators)
- Number of features taken during each split(max_features)

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
