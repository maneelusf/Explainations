## About
This repository contains the implementation of Naive Bayes. To use this library, call the NaiveBayes621 class from the 
bayes.py file.

## Example
```sh
from bayes import *
V, X, y = training_data()
d1 = vectorize(V, words("very good, the story is xyzdef appealing. i also try to recommend excellent films like this"))
d2 = vectorize(V, words("brexit vote postponed hated movie; a van damme movie has become a painful chore"))
y_test = np.array([1, 0])
X_test = np.vstack([d1, d2])
model = NaiveBayes621()
model.fit(X, y)
y_pred = model.predict(X_test)
```