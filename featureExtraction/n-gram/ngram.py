import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def ngram(filepath, n):
    df = pd.read_csv(filepath, sep=";;;", engine="python")
    cv = CountVectorizer(ngram_range=(1, n), max_features=2500)
    unigram_matrix = cv.fit_transform(df["teks"].tolist())
    feat_name = cv.get_feature_names()

    return unigram_matrix, feat_name
