import nltk
import pandas as pd
from nltk import CRFTagger
from collections import Counter
from sklearn.model_selection import train_test_split

def run_postag(filename):
    df = pd.read_csv(filename, delimiter=";", low_memory=False, header=0)

    ct = CRFTagger()
    ct.set_model_file("all_indo_man_tag_corpus_model.crf.tagger")
    pos_feat_list = []
    count_tag = []
    for tweet in df["teks"].tolist():
        token = eval(tweet)
        tag = ct.tag_sents([token])
        flat_tag = [item for sublist in tag for item in sublist]
        pos_count = Counter([j for i, j in flat_tag])
        pos_feat = (pos_count['JJ'], pos_count['NEG'], pos_count['RB'], pos_count['UH'])
        pos_feat_list.append(pos_feat)

    return pos_feat_list