import time
import datetime
import warnings
import numpy as np
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, train_test_split, cross_val_score


def decision_tree(result_fe, label):
    cv = KFold(n_splits=20, random_state=1, shuffle=True)
    x_train, x_test, y_train, y_test = train_test_split(result_fe, label)
    
    dectree = DecisionTreeClassifier(random_state=1)
    dectree.fit(x_train, y_train)
    predictions = dectree.predict(x_test)

    scores = cross_val_score(dectree, result_fe, label, scoring='f1_micro', cv=cv, n_jobs=-1)
    print('F1-score: %.3f (%.3f)' % (mean(scores), std(scores)))
    return mean(scores)