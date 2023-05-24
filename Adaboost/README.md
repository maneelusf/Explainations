## About
This repository contains the implementation of the Adaboost Algorithm. To use this library, call the adaboost function from the 
adaboost.py. For predictions, use the adaboost_predict function.

## Example
```sh
from adaboost import *
X, Y = parse_spambase_data(PATH/"spambase.train")
X_test, Y_test = parse_spambase_data(PATH/"spambase.test")
trees, trees_weights = adaboost(X, Y, 10)
Yhat = adaboost_predict(X, trees, trees_weights)
Yhat_test = adaboost_predict(X_test, trees, trees_weights)
    
acc_test = accuracy(Y_test, Yhat_test)
acc_train = accuracy(Y, Yhat)
print("Train Accuracy %.4f" % acc_train)
print("Test Accuracy %.4f" % acc_test)
```
