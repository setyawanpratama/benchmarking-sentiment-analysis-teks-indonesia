import time
import datetime
import numpy as np
import pandas as pd
from gensim.models import FastText


def create_fasttext_cbow_model(filepath, vector_size):
    start_time = time.time()
    model_cbow = FastText(data, min_count=1, window=5, vector_size=vector_size)
    model_cbow.save('./featureExtraction/fasttext/models_all/model_cbow_{}.model'.format(vector_size))
    end_time = time.time()
    print("CBOW Time: {} seconds".format(end_time - start time))
    
    return model_cbow


def create_fasttext_sg_model(filepath, vector_size):
    start_time = time.time()
    model_sg_100 = FastText(data, min_count=1, window=5, sg=1, vector_size=vector_size)
    model_sg_100.save('./featureExtraction/fasttext/models_all/model_sg_{}.model'.format(vector_size))
    end_time = time.time()
    print("SG 100 Time: {} seconds".format(end_time - start_time))


def fasttext(filepath, cbow, vector_size):
    df = pd.read_csv(filepath, delimiter=";", low_memory=False, header=0)
    model_path = './featureExtraction/fasttext/models_all/model_{}_{}.model'.format('cbow' if cbow == 1 else 'sg', vector_size)
    model = FastText.load(model_path)

    word2vec_arr=[]
    for row in df['teks'].tolist():
        tweets = row.split(" ")
        row_mean_vector = (np.mean([model_cbow.wv[terms] for terms in tweets], axis=0)).tolist()
        if not (type(row_mean_vector) is list):
            row_mean_vector = [float(0) for i in range(vector_size)]
        word2vec_arr.append(row_mean_vector)

    return np.array(word2vec_arr), df['label'].tolist()
