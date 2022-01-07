import numpy as np
import pandas as pd

def run_ortografi(filepath):
    df = pd.read_csv(filepath, sep="\;\;\;", engine="python", header=0, index_col=False)
    df.dropna(axis=0, inplace=True)

    features = ["char_len", "word_len", "symbol", "upper"]
    all_orto_feat = []
    for tweet in df["teks"].tolist():
        char_len = len(tweet)
        word_len = len(tweet.split(" "))
        symbol = sum((1 for c in tweet if not c.isalnum()))
        upper = sum((1 for c in tweet if c.isupper()))
        orto_feat = [char_len, word_len, symbol, upper]
        all_orto_feat.append(orto_feat)

    return np.array(all_orto_feat), df["label"].tolist(), features
