import numpy as np
import pandas as pd
from gensim.models import Word2Vec


def ortho_w2v(filepath, hs, neg, vector_size):
    # w2v
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
    
    # W2V
    model_path = 'featureExtraction/word2vec/models_all/model_sg_{}_{}.model'.format('hs' if hs == 1 else 'neg', vector_size)
    model_sg = Word2Vec.load(model_path)

    word2vec_arr=[]
    for row in df['teks1'].tolist():
        tweets = row.split(" ")
        row_mean_vector = (np.mean([model_sg.wv[terms] for terms in tweets], axis=0)).tolist()
        if not (type(row_mean_vector) is list):
            row_mean_vector = [float(0) for i in range(vector_size)]
        word2vec_arr.append(row_mean_vector)

    
    # Ortho
    features = ["char_len", "word_len", "symbol", "upper"]
    all_orto_feat = []
    for tweet in df["teks2"].tolist():
        char_len = len(tweet)
        word_len = len(tweet.split(" "))
        symbol = sum((1 for c in tweet if not c.isalnum()))
        upper = sum((1 for c in tweet if c.isupper()))
        orto_feat = [char_len, word_len, symbol, upper]
        all_orto_feat.append(orto_feat)

    return np.hstack((np.array(all_orto_feat), np.array(word2vec_arr))), df["label1"].tolist(), features
