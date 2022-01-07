import numpy as np
import pandas as pd
from nltk import CRFTagger
from collections import Counter


def ortho_pos(filepath):
    # tf
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
    
    # POS
    ct = CRFTagger()
    ct.set_model_file("featureExtraction/pos/all_indo_man_tag_corpus_model.crf.tagger")
    pos_feat_list = []
    count_tag = []
    for tweet in df["teks1"].tolist():
        token = tweet.split(" ")
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

    return np.hstack((np.array(all_orto_feat), np.array(pos_feat_list))), df["label1"].tolist(), features
