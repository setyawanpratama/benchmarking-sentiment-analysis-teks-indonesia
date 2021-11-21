import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


def ngram(filepath, n):
    df = pd.read_csv(filepath, sep="\;\;\;", engine="python", header=0, index_col=False)
    df.dropna(axis=0, inplace=True)
    cv = CountVectorizer(ngram_range=(n, n))
    unigram_matrix = cv.fit_transform(df["teks"].tolist())
    feat_name = cv.get_feature_names()

    return unigram_matrix.toarray(), feat_name
