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
    # return df_hasil

    return np.array(fitur_sentimen_all), emosi


def run_lexiconID_tweet(filename):
    CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

    pos = pd.read_csv(os.path.join(CURRENT_DIR, "lexicon-source/ID-OpinionWords/positive.txt"), header=None, names=["pos"])
    list_pos = pos["pos"].tolist()
    neg = pd.read_csv(os.path.join(CURRENT_DIR, "lexicon-source/ID-OpinionWords/negative.txt"), header=None, names=["neg"])
    list_neg = neg["neg"].tolist()

    df = pd.read_csv(filename, delimiter=";", low_memory=False, header=0)
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

    return fitur_sentimen_all, emosi


def run_lexiconInset_tweet(filename):
    CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

    pos = pd.read_csv(os.path.join(CURRENT_DIR, "lexicon-source/InSet/positive.tsv"), header=0, sep="\t")
    list_pos = pos["word"].tolist()
    dict_pos = dict(zip(pos.word, pos.weight))
    neg = pd.read_csv(os.path.join(CURRENT_DIR, "lexicon-source/InSet/negative.tsv"), header=0, sep="\t")
    list_neg = neg["word"].tolist()
    dict_neg = dict(zip(neg.word, neg.weight))

    df = pd.read_csv(filename, delimiter=";", low_memory=False, header=0)
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
                emosi_value["positif"] += int(dict_pos[k])
            if k in list_neg:
                emosi_value["negatif"] -= int(dict_neg[k])

        fitur_sentimen_perkalimat = list(emosi_value.values())
        fitur_sentimen_all.append(fitur_sentimen_perkalimat)

    # df_hasil = pd.DataFrame(fitur_sentimen_all, columns=emosi)

    return np.array(fitur_sentimen_all), emosi
