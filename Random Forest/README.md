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
from rf import RandomForestRegressor621, RandomForestClassifier621
X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
n_estimators,min_samples_leaf,max_features,oob = 15,3,0.3,False
rf = RandomForestRegressor621(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, max_features=max_features, oob_score=oob)
dt.fit(X_train, y_train)
score = dt.score(X_test, y_test)
```
