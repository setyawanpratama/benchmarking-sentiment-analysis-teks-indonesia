import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def run_tfidf_tweet(filepath):
    df = pd.read_csv(filepath, delimiter=";", low_memory=False, header=0)
    df.dropna(axis=0, inplace=True)
    vectorizer = TfidfVectorizer()
    tfidf_feature = vectorizer.fit_transform(df['teks'].tolist())

    return tfidf_feature.toarray(), df["label"].tolist()
