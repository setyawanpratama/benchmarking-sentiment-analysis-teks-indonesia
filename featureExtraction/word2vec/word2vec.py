import numpy as np
import pandas as pd
from gensim.models import Word2Vec


def word2vec_cbow(filepath, hierarchial_softmax, negative, vector_size):
    df = pd.read_csv(filepath, delimiter=";", low_memory=False, header=0)

    data = []
    for tweet in df["teks"].tolist():
        list_kata = eval(tweet)
        data.append(list_kata)

    model_cbow = Word2Vec(data, min_count=1, window=5, hs=hierarchial_softmax, negative=negative, vector_size=vector_size)
    return model_cbow

def word2vec_sg(filepath, hierarchial_softmax, negative, vector_size):
    df = pd.read_csv(filepath, delimiter=";", low_memory=False, header=0)

    data = []
    for tweet in df["teks"].tolist():
        list_kata = eval(tweet)
        data.append(list_kata)

    model_skipgram = Word2Vec(data, min_count=1, window=5, sg=1, hs=hierarchial_softmax, negative=negative, vector_size=vector_size)
    return model_skipgram