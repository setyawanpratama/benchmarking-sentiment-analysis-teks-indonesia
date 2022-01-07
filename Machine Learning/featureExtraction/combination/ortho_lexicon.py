import numpy as np
import pandas as pd


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
    
    # lexicon
    pos1 = pd.read_csv("featureExtraction/lexicon/lexicon-source/vania/positive.txt", header=None, names=["pos"])
    list_pos1 = pos1["pos"].tolist()
    neg1 = pd.read_csv("featureExtraction/lexicon/lexicon-source/vania/negative.txt", header=None, names=["neg"])
    list_neg1 = neg1["neg"].tolist()

    pos2 = pd.read_csv("featureExtraction/lexicon/lexicon-source/InSet/positive.tsv", header=0, sep="\t")
    list_pos2 = pos2["word"].tolist()
    dict_pos2 = dict(zip(pos2.word, pos2.weight))
    neg2 = pd.read_csv("featureExtraction/lexicon/lexicon-source/InSet/negative.tsv", header=0, sep="\t")
    list_neg2 = neg2["word"].tolist()
    dict_neg2 = dict(zip(neg2.word, neg2.weight))

    emosi = ["positif_vania", "negatif_vania", "positif_inset", "negatif_inset"]
    fitur_sentimen_all = []

    for tweet in df["teks1"].tolist():
        # inisiasi value
        value = [0, 0, 0, 0]
        emosi_value = {}
        for i in range(len(emosi)):
            emosi_value[emosi[i]] = value[i]

        list_kata = tweet.split(" ")
        for k in list_kata:
            if k in list_pos1:
                emosi_value["positif_vania"] += 1
            if k in list_neg1:
                emosi_value["negatif_vania"] += 1
            if k in list_pos2:
                emosi_value["positif_inset"] += int(dict_pos2[k])
            if k in list_neg2:
                emosi_value["negatif_inset"] -= int(dict_neg2[k])

        fitur_sentimen_perkalimat = list(emosi_value.values())
        fitur_sentimen_all.append(fitur_sentimen_perkalimat)

    
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

    return np.hstack((np.array(all_orto_feat), np.array(fitur_sentimen_all))), df["label1"].tolist(), features
