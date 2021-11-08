# evaluate a logistic regression model using k-fold cross-validation
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

def kFold(data):
    # create dataset
    X, y = make_classification(n_samples=len(data), n_features=len(data[0]), random_state=1)
    # prepare the cross-validation procedure
    cv = KFold(n_splits=10, random_state=1, shuffle=True)