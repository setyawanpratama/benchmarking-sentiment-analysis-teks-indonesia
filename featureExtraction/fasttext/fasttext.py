import numpy as np
import pandas as pd
from gensim.models import FastText


def fasttext(filepath, vector_size):
    df = pd.read_csv(filepath, delimiter=";", low_memory=False, header=0)

    data = []
    for tweet in df["teks"].tolist():
        list_kata = eval(tweet)
        data.append(list_kata)

    model_fasttext = FastText(data, min_count=1, window=5, vector_size=vector_size)
    fasttext_arr = []
    for tweet in data:
        row_mean_vector = (np.mean([model_fasttext.wv[terms] for terms in tweet], axis=0)).tolist()
        if not (type(row_mean_vector) is list):
            row_mean_vector = [float(0) for i in range(vector_size)]
        fasttext_arr.append(row_mean_vector)
    
    return model_fasttext, np.array(fasttext_arr)


def fasttextsg(filepath, vector_size):
    df = pd.read_csv(filepath, delimiter=";", low_memory=False, header=0)

    data = []
    for tweet in df["teks"].tolist():
        list_kata = eval(tweet)
        data.append(list_kata)

    model_fasttext_skipgram = FastText(data, min_count=1, window=5, sg=1, vector_size=vector_size)
    fasttext_arr = []
    for tweet in data:
        row_mean_vector = (np.mean([model_fasttext_skipgram.wv[terms] for terms in tweet], axis=0)).tolist()
        if not (type(row_mean_vector) is list):
            row_mean_vector = [float(0) for i in range(vector_size)]
        fasttext_arr.append(row_mean_vector)

    return model_fasttext_skipgram, np.array(fasttext_arr)
