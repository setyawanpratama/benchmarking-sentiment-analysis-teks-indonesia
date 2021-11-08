import numpy as np
import pandas as pd
import matplotlib
import seaborn
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def run_tfidf_tweet(filename):
    df = pd.read_csv(filename, delimiter=";;", low_memory=False, header=)

    x_train, x_test, y_train, y_test = train_test_split(df['tweet'], df['label'], test_size=0.25)
    print("Train: ", x_train.shape, y_train.shape, "\nTest: ", x_test.shape, y_test.shape)

    tfidfconverter = TfidfVectorizer(max_features=2500 ,min_df=3, max_df=0.75)
    tfidf_x_train = tfidfconverter.fit_transform(x_train)
    tfidf_x_test = tfidfconverter.transform(x_test)
