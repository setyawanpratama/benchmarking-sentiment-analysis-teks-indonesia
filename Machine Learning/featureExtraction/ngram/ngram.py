import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


def ngram(filepath, n):
    df = pd.read_csv(filepath, delimiter=";", low_memory=False, header=0)
    df.dropna(axis=0, inplace=True)
    
    cv = CountVectorizer(ngram_range=(n, n))
    ngram_matrix = cv.fit_transform(df["teks"].tolist())
    feat_name = cv.get_feature_names()

    return ngram_matrix.toarray(), df["label"].tolist(), feat_name
