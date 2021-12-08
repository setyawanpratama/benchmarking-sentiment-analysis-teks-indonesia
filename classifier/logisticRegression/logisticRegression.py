import time
import datetime
import warnings
from numpy import mean
from numpy import std
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score


def logistic_regression(result_fe, label):
    cv = KFold(n_splits=20, random_state=1, shuffle=True)
    logRes = LogisticRegression(random_state=1, max_iter=10000)

    scores = cross_val_score(logRes, result_fe, label, scoring="f1_micro", cv=cv, n_jobs=-1)
    print("F1-score: %.3f (%.3f)" % (mean(scores), std(scores)))
    return mean(scores)
