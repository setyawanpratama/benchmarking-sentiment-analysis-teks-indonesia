import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def term_freq(filepath, n):
    df = pd.read_csv(filepath, sep="\;\;\;", engine="python", header=0, index_col=False)
    df.dropna(axis=0, inplace=True)
    cv = CountVectorizer(ngram_range=(1, n))
    tf_matrix = cv.fit_transform(df["teks"].tolist())

    return tf_matrix.toarray(), df["label"].tolist()
