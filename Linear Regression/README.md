## About
This repository contains the implementation of Linear,Logistic & Ridge Regression. To use this library, call the LinearRegression621(For Regressor), LogisticRegression621(for Classifier) & RidgeRegression621(for Ridge) from the 
linreg.py file.

## Example
```sh
from linreg import *
from sklearn.datasets import load_boston
boston = load_boston()
X = boston.data
y = boston.target
y = y.reshape(-1, 1)
model = LinearRegression621(max_iter=15_000, eta=5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, shuffle=True)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
```