from sklearn.linear_model import LogisticRegression
from sklearn import metrics

def logistic_regression(features_list, x_train, x_test, y_train, y_test):
    logRes = LogisticRegression()
    logRes.fit(x_train, y_train)
    predictions = logRes.predict(x_test)
    
    score = logRes.score(x_test, y_test)
    f1_score = metrics.f1_score(y_test, predictions)


    return (f1_score, score)