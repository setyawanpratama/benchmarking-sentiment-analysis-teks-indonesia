import os
import h5py
from nltk.util import usage
import numpy as np
import datetime
import time
import warnings

from featureExtraction.tf.tf import term_freq
from featureExtraction.tfidf.tfidf import run_tfidf_tweet
from featureExtraction.fasttext.fasttext import fasttext, fasttextsg
from featureExtraction.lexicon.lexicon import run_lexiconVania_tweet, run_lexiconInset_tweet, run_lexiconCombined_tweet
from featureExtraction.orthography.orthography import run_ortografi
from featureExtraction.pos.postag import run_postag
from featureExtraction.tf.tf import term_freq
from featureExtraction.word2vec.word2vec import word2vec_cbow, word2vec_sg
from featureExtraction.ngram.ngram import ngram

from featureSelection.kFold import kFold

from logger import CSVLogger
from time_counter import Timer


warnings.filterwarnings("ignore")


def main(DATA_TEMPLATED, DATA_CLEAN):
    program_time = Timer("Program")
        
    TARGET_DIR = "featureExtractionResults\\run-" + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    NEW_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), TARGET_DIR)
    os.mkdir(NEW_DIR)
    os.mkdir(os.path.join(NEW_DIR, "models"))

    logger = CSVLogger(NEW_DIR + "/log.csv")
    logger.do_log(["Description","hours","minutes","seconds","total"])
    

    program_time.stop()

    pass


def print_info(item, dataset):
    print("Running: {:<20} --> {:<100}".format(item, dataset), end="\r")


if __name__ == "__main__":
    templated = [x for x in os.listdir("./Dataset/templated")]
    templated.sort()
    clean = [x for x in os.listdir("./Dataset/clean")]
    clean.sort()

    DATA_TEMPLATED = {}
    DATA_CLEAN = {}

    for i in range(len(templated) - 1):
        DATA_TEMPLATED[i] = ["./Dataset/templated/" + templated[i], templated[i]]
    for i in range(len(clean) - 1):
        DATA_CLEAN[i] = ["./Dataset/clean/" + clean[i], clean[i]]

    main(DATA_TEMPLATED, DATA_CLEAN)
