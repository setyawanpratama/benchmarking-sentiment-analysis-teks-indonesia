import nltk
import os
import numpy as np
import pandas as pd
from nltk import CRFTagger
from collections import Counter
from sklearn.model_selection import train_test_split


def run_postag(filename):
    CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
    df = pd.read_csv(filename, delimiter=";", low_memory=False, header=0)
    df.dropna(axis=0, inplace=True)

    ct = CRFTagger()
    ct.set_model_file(
        os.path.join(CURRENT_DIR, "all_indo_man_tag_corpus_model.crf.tagger")
    )
    pos_feat_list = []
    count_tag = []
    for tweet in df["teks"].tolist():
        token = eval(tweet)
        if len(token) < 1:
            pos_feat = [0, 0, 0, 0]
            pos_feat_list.append(pos_feat)
            continue
        tag = ct.tag_sents([token])
        flat_tag = [item for sublist in tag for item in sublist]
        pos_count = Counter([j for i, j in flat_tag])
        pos_feat = [
            pos_count["JJ"] if not np.isnan(pos_count["JJ"]) else 0,
            pos_count["NEG"] if not np.isnan(pos_count["NEG"]) else 0,
            pos_count["RB"] if not np.isnan(pos_count["RB"]) else 0,
            pos_count["UH"] if not np.isnan(pos_count["UH"]) else 0,
        ]
        pos_feat_list.append(pos_feat)

    return np.array(pos_feat_list), df["label"].tolist()
