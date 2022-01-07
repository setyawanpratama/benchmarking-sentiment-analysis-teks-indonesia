import time
import datetime
import warnings
import numpy as np
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, train_test_split, cross_val_score


def decision_tree(result_fe, label):
    cv = KFold(n_splits=20, random_state=1, shuffle=True)    
    dectree = DecisionTreeClassifier(random_state=1)

    acc = cross_val_score(dectree, result_fe, label, scoring='f1_micro', cv=cv, n_jobs=-1)
    f1 = cross_val_score(dectree, result_fe, label, scoring='f1_macro', cv=cv, n_jobs=-1)
    return acc, f1
