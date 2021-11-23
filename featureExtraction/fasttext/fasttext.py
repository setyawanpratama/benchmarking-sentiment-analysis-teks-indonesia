from gensim import models
import numpy as np
import pandas as pd
from gensim.models import FastText


def fasttext(filepath):
    df = pd.read_csv(filepath, delimiter=";", low_memory=False, header=0)

    data = []
    for tweet in df["teks"].tolist():
        list_kata = eval(tweet)
        data.append(list_kata)

    model_fasttext = FastText(data, min_count=1, window=5)
    fasttext_arr=[]
    for tweet in data:
        row_mean_vector = (np.mean([model_fasttext.wv[terms] for terms in tweet], axis=0)).tolist()
        fasttext_arr.append(row_mean_vector)

    return model_fasttext, fasttext_arr

def fasttextsg(filepath):
    df = pd.read_csv(filepath, delimiter=";", low_memory=False, header=0)

    data = []
    for tweet in df["teks"].tolist():
        list_kata = eval(tweet)
        data.append(list_kata)

    model_fasttext_skipgram = FastText(data, min_count=1, window=5, sg=1)
    fasttext_arr=[]
    for tweet in data:
        row_mean_vector = (np.mean([model_fasttext_skipgram.wv[terms] for terms in tweet], axis=0)).tolist()
        fasttext_arr.append(row_mean_vector)

    return model_fasttext_skipgram, fasttext_arr