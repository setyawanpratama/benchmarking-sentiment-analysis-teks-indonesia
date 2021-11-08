import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def run_tfidf_tweet(filename):
    df = pd.read_csv(filename, sep=";;;", engine='python')

    vectorizer = TfidfVectorizer()
    tfidf_feature = vectorizer.fit_transform(df['teks'].tolist())

    return tfidf_feature