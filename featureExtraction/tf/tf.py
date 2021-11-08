import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def tf(filepath, n):
    df = pd.read_csv(filepath, sep="\;\;\;", engine="python")
    cv = CountVectorizer(ngram_range=(1, 1))
    tf_matrix = cv.fit_transform(df["teks"].tolist())

    return tf_matrix
