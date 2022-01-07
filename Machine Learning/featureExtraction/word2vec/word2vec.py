import time
import datetime
import numpy as np
import pandas as pd
from gensim.models import Word2Vec


def create_word2vec_cbow_model(data, hs, neg, vector_size):
    start_time = time.time()
    model_cbow = Word2Vec(data, min_count=1, window=5, hs=hs, negative=neg, vector_size=vector_size)
    model_cbow.save('./featureExtraction/word2vec/models_all/model_cbow_{}_{}.model'.format('hs' if hs == 1 else 'neg', vector_size))
    end_time = time.time()
    print("Time: {} seconds".format(end_time - start_time))
    
    return model_cbow


def create_word2vec_sg_model(data, hs, neg, vector_size):
    start_time = time.time()
    model_sg = Word2Vec(data, min_count=1, sg=1, window=5, hs=hs, negative=neg, vector_size=vector_size)
    model_sg.save('./featureExtraction/word2vec/models_all/model_sg_{}_{}.model'.format('hs' if hs == 1 else 'neg', vector_size))
    end_time = time.time()
    print("Time: {} seconds".format(end_time - start_time))
    
    return model_sg


def word2vec_cbow(filepath, hs, neg, vector_size):
    df = pd.read_csv(filepath, delimiter=";", low_memory=False, header=0)
    model_path = './featureExtraction/word2vec/models_all/model_cbow_{}_{}.model'.format('hs' if hs == 1 else 'neg', vector_size)
    model_cbow = Word2Vec.load(model_path)

    word2vec_arr=[]
    for row in df['teks'].tolist():
        tweets = row.split(" ")
        row_mean_vector = (np.mean([model_cbow.wv[terms] for terms in tweets], axis=0)).tolist()
        if not (type(row_mean_vector) is list):
            row_mean_vector = [float(0) for i in range(vector_size)]
        word2vec_arr.append(row_mean_vector)

    return np.array(word2vec_arr), df['label'].tolist()

def word2vec_sg(filepath, hs, vector_size):
    df = pd.read_csv(filepath, delimiter=";", low_memory=False, header=0)
    model_path = './featureExtraction/word2vec/models_all/model_sg_{}_{}.model'.format('hs' if hs == 1 else 'neg', vector_size)
    model_sg = Word2Vec.load(model_path)

    word2vec_arr=[]
    for row in df['teks'].tolist():
        tweets = row.split(" ")
        row_mean_vector = (np.mean([model_sg.wv[terms] for terms in tweets], axis=0)).tolist()
        if not (type(row_mean_vector) is list):
            row_mean_vector = [float(0) for i in range(vector_size)]
        word2vec_arr.append(row_mean_vector)

    return np.array(word2vec_arr), df['label'].tolist()
