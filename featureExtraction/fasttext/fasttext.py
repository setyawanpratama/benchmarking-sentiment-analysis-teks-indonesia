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
    return model_fasttext

def fasttextsg(filepath):
    df = pd.read_csv(filepath, delimiter=";", low_memory=False, header=0)

    data = []
    for tweet in df["teks"].tolist():
        list_kata = eval(tweet)
        data.append(list_kata)

    model_fasttext_skipgram = FastText(data, min_count=1, window=5, sg=1)

    return model_fasttext_skipgram