import os
import numpy as np
import pandas as pd

LEXICON_LIST = ["ID-OpinionWords", "InSet", "vania"]

def run_lexiconVania_tweet(filename):
    CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

    pos = pd.read_csv(os.path.join(CURRENT_DIR, "lexicon-source/vania/positive.txt"), header=None, names=["pos"])
    list_pos = pos["pos"].tolist()
    neg = pd.read_csv(os.path.join(CURRENT_DIR, "lexicon-source/vania/negative.txt"), header=None, names=["neg"])
    list_neg = neg["neg"].tolist()

    df = pd.read_csv(filename, delimiter=";", low_memory=False, header=0)
    df.dropna(axis=0, inplace=True)

    emosi = ["positif", "negatif"]
    fitur_sentimen_all = []

    for tweet in df["teks"].tolist():
        # inisiasi value
        value = [0, 0]
        emosi_value = {}
        for i in range(len(emosi)):
            emosi_value[emosi[i]] = value[i]

        list_kata = tweet.split(" ")
        for k in list_kata:
            if k in list_pos:
                emosi_value["positif"] += 1
            if k in list_neg:
                emosi_value["negatif"] += 1

        fitur_sentimen_perkalimat = list(emosi_value.values())
        fitur_sentimen_all.append(fitur_sentimen_perkalimat)

    return np.array(fitur_sentimen_all), df["label"].tolist(), emosi


def run_lexiconID_tweet(filename):
    CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

    pos = pd.read_csv(os.path.join(CURRENT_DIR, "lexicon-source/ID-OpinionWords/positive.txt"), header=None, names=["pos"])
    list_pos = pos["pos"].tolist()
    neg = pd.read_csv(os.path.join(CURRENT_DIR, "lexicon-source/ID-OpinionWords/negative.txt"), header=None, names=["neg"])
    list_neg = neg["neg"].tolist()

    df = pd.read_csv(filename, delimiter=";", low_memory=False, header=0)
    df.dropna(axis=0, inplace=True)
    # new_header = df.iloc[0] #grab the first row for the header
    # df = df[1:] #take the data less the header row
    # df.columns = new_header #set the header row as the df header

    emosi = ["positif", "negatif"]
    fitur_sentimen_all = []

    for tweet in df["teks"].tolist():
        # inisiasi value
        value = [0, 0]
        emosi_value = {}
        for i in range(len(emosi)):
            emosi_value[emosi[i]] = value[i]

        list_kata = eval(tweet)
        for k in list_kata:
            if k in list_pos:
                emosi_value["positif"] += 1
            if k in list_neg:
                emosi_value["negatif"] += 1

        fitur_sentimen_perkalimat = list(emosi_value.values())
        fitur_sentimen_all.append(fitur_sentimen_perkalimat)

    # df_hasil = pd.DataFrame(fitur_sentimen_all, columns=emosi)

    return fitur_sentimen_all, df["label"].tolist(), emosi


def run_lexiconInset_tweet(filename):
    CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

    pos = pd.read_csv(os.path.join(CURRENT_DIR, "lexicon-source/InSet/positive.tsv"), header=0, sep="\t")
    list_pos = pos["word"].tolist()
    dict_pos = dict(zip(pos.word, pos.weight))
    neg = pd.read_csv(os.path.join(CURRENT_DIR, "lexicon-source/InSet/negative.tsv"), header=0, sep="\t")
    list_neg = neg["word"].tolist()
    dict_neg = dict(zip(neg.word, neg.weight))

    df = pd.read_csv(filename, delimiter=";", low_memory=False, header=0)
    df.dropna(axis=0, inplace=True)
    # new_header = df.iloc[0] #grab the first row for the header
    # df = df[1:] #take the data less the header row
    # df.columns = new_header #set the header row as the df header

    emosi = ["positif", "negatif"]
    fitur_sentimen_all = []

    for tweet in df["teks"].tolist():
        # inisiasi value
        value = [0, 0]
        emosi_value = {}
        for i in range(len(emosi)):
            emosi_value[emosi[i]] = value[i]

        list_kata = tweet.split(" ")
        for k in list_kata:
            if k in list_pos:
                emosi_value["positif"] += int(dict_pos[k])
            if k in list_neg:
                emosi_value["negatif"] -= int(dict_neg[k])

        fitur_sentimen_perkalimat = list(emosi_value.values())
        fitur_sentimen_all.append(fitur_sentimen_perkalimat)

    # df_hasil = pd.DataFrame(fitur_sentimen_all, columns=emosi)

    return np.array(fitur_sentimen_all), df["label"].tolist(), emosi

def run_lexiconCombined_tweet(filename):
    CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
    
    pos1 = pd.read_csv(os.path.join(CURRENT_DIR, "lexicon-source/vania/positive.txt"), header=None, names=["pos"])
    list_pos1 = pos1["pos"].tolist()
    neg1 = pd.read_csv(os.path.join(CURRENT_DIR, "lexicon-source/vania/negative.txt"), header=None, names=["neg"])
    list_neg1 = neg1["neg"].tolist()

    pos2 = pd.read_csv(os.path.join(CURRENT_DIR, "lexicon-source/InSet/positive.tsv"), header=0, sep="\t")
    list_pos2 = pos2["word"].tolist()
    dict_pos2 = dict(zip(pos2.word, pos2.weight))
    neg2 = pd.read_csv(os.path.join(CURRENT_DIR, "lexicon-source/InSet/negative.tsv"), header=0, sep="\t")
    list_neg2 = neg2["word"].tolist()
    dict_neg2 = dict(zip(neg2.word, neg2.weight))


    df = pd.read_csv(filename, delimiter=";", low_memory=False, header=0)
    df.dropna(axis=0, inplace=True)

    emosi = ["positif_vania", "negatif_vania", "positif_inset", "negatif_inset"]
    fitur_sentimen_all = []

    for tweet in df["teks"].tolist():
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

    return np.array(fitur_sentimen_all), df["label"].tolist(), emosi