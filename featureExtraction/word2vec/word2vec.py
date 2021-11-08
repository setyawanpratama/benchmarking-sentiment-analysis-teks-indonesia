import numpy as np
import pandas as pd
from gensim.models import Word2Vec


def word2vec(filepath):
    df = pd.read_csv(filepath, delimiter=";", low_memory=False, header=0)

    data = []
    for tweet in df["teks"].tolist():
        list_kata = eval(tweet)
        data.append(list_kata)

    # print(data[:5])
    model_cbow = Word2Vec(data, min_count=1, window=5)
    model_skipgram = Word2Vec(data, min_count=1, window=5, sg=1)

    # print(model_cbow)
    model_cbow.save("./models/model_cbow.model")
    model_skipgram.save("./models/model_skipgram.model")

    return model_cbow, model_skipgram

def word2vec_corpus(filepath):
    