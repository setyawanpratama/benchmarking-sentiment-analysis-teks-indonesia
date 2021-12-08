import time
import datetime
import warnings
from numpy import mean
from numpy import std
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold, cross_val_score

def multinomial_nb(result_fe, label):
    cv = KFold(n_splits=20, random_state=1, shuffle=True)
    multiNB = MultinomialNB()

    scores = cross_val_score(multiNB, result_fe, label, scoring='f1_micro', cv=cv, n_jobs=-1)
    print('F1-score: %.3f (%.3f)' % (mean(scores), std(scores)))
    return mean(scores)