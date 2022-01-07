import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def ortho_tf(filepath):
    # tf
    clean = "../Dataset/Clean/{}".format(filepath)
    df1 = pd.read_csv(clean, delimiter=";", low_memory=False, header=0)
    df1.rename(columns={"teks": "teks1", "label": "label1"}, inplace=True)
    # ortho
    templated = "../Dataset/Raw/{}".format(filepath)
    df2 = pd.read_csv(templated, sep="\;\;\;", engine="python", header=0, index_col=False)
    df2.rename(columns={"teks": "teks2", "label": "label2"}, inplace=True)
    # join
    df = pd.concat([df1, df2], axis=1)
    df.dropna(axis=0, inplace=True)
    
    cv = CountVectorizer(ngram_range=(1, 1))
    tf_matrix = cv.fit_transform(df["teks1"].tolist())
    
    features = ["char_len", "word_len", "symbol", "upper"]
    all_orto_feat = []
    for tweet in df["teks2"].tolist():
        char_len = len(tweet)
        word_len = len(tweet.split(" "))
        symbol = sum((1 for c in tweet if not c.isalnum()))
        upper = sum((1 for c in tweet if c.isupper()))
        orto_feat = [char_len, word_len, symbol, upper]
        all_orto_feat.append(orto_feat)

    return np.hstack((np.array(all_orto_feat), tf_matrix.toarray())), df["label1"].tolist(), features
