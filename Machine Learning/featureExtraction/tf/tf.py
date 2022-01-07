import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def term_freq(filepath):
    df = pd.read_csv(filepath, delimiter=";", low_memory=False, header=0)
    df.dropna(axis=0, inplace=True)
    
    cv = CountVectorizer(ngram_range=(1, 1))
    tf_matrix = cv.fit_transform(df["teks"].tolist())

    return tf_matrix.toarray(), df["label"].tolist()
