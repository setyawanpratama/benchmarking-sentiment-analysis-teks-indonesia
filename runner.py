import os
import h5py
from nltk.util import usage
import numpy as np
import datetime
import time

from featureExtraction.tf.tf import *
from featureExtraction.tfidf.tfidf import *
from featureExtraction.fasttext.fasttext import *
from featureExtraction.lexicon.lexicon import *
from featureExtraction.orthography.orthography import *
from featureExtraction.pos.postag import *
from featureExtraction.tf.tf import *
from featureExtraction.word2vec.word2vec import *
from featureExtraction.ngram.ngram import *

import featureSelection as featSel

# ; = template
# ;;; = templated


TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
TARGET_DIR = "featureExtractionResults/run-" + TIMESTAMP
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
NEW_DIR = os.path.join(CURRENT_DIR, TARGET_DIR)
os.mkdir(NEW_DIR)


def main(hf, templated, clean, raw):
    init_time = time.time()
    for i in templated:
        result_unigram, feat_name = ngram(i, 1)
        hf.create_dataset("unigram" + i, data=result_unigram)
    end_time = time.time()
    get_time_diff(init_time, end_time, "FE-Unigram")

    init_time = time.time()
    for i in templated:
        result_tf = term_freq(i, 1)
        hf.create_dataset("term_freq" + i, data=result_tf)
    end_time = time.time()
    get_time_diff(init_time, end_time, "FE-TF")

    init_time = time.time()
    for i in templated:
        result_tf_idf = run_tfidf_tweet(i, 1)
        hf.create_dataset("tf_idf" + i, data=result_tf_idf)
    end_time = time.time()
    get_time_diff(init_time, end_time, "FE-TFIDF")

    init_time = time.time()
    for i in clean:
        result_lexiconVania = run_lexiconVania_tweet(i)
        hf.create_dataset("lex_vania" + i, data=result_lexiconVania)
    end_time = time.time()
    get_time_diff(init_time, end_time, "FE-Lexicon Vania")

    init_time = time.time()
    for i in clean:
        result_lexiconInSet = run_lexiconInset_tweet(i)
        hf.create_dataset("lex_inset" + i, data=result_lexiconInSet)
    end_time = time.time()
    get_time_diff(init_time, end_time, "FE-Lexicon InSet")

    init_time = time.time()
    for i in clean:
        result_fasttext = fasttext(i)
        hf.create_dataset("fasttext" + i, data=result_fasttext)
    end_time = time.time()
    get_time_diff(init_time, end_time, "FE-FastText")

    init_time = time.time()
    for i in clean:
        result_w2vhs100 = word2vec(i, 0, 5, 100)
        hf.create_dataset("w2vhs100" + i, data=result_w2vhs100)
    end_time = time.time()
    get_time_diff(init_time, end_time, "FE-Word2Vec HS Vec100")

    init_time = time.time()
    for i in clean:
        result_w2vneg100 = word2vec(i, 0, 5, 100)
        hf.create_dataset("w2vneg100" + i, data=result_w2vneg100)
    end_time = time.time()
    get_time_diff(init_time, end_time, "FE-Word2Vec NEG Vec100")

    init_time = time.time()
    for i in clean:
        result_w2vhs200 = word2vec(i, 0, 5, 200)
        hf.create_dataset("w2vhs200" + i, data=result_w2vhs200)
    end_time = time.time()
    get_time_diff(init_time, end_time, "FE-Word2Vec HS Vec200")

    init_time = time.time()
    for i in clean:
        result_w2vneg200 = word2vec(i, 0, 5, 200)
        hf.create_dataset("w2vneg200" + i, data=result_w2vneg200)
    end_time = time.time()
    get_time_diff(init_time, end_time, "FE-Word2Vec NEG Vec200")

    init_time = time.time()
    for i in clean:
        result_w2vhs300 = word2vec(i, 0, 5, 300)
        hf.create_dataset("w2vhs300" + i, data=result_w2vhs300)
    end_time = time.time()
    get_time_diff(init_time, end_time, "FE-Word2Vec HS Vec300")

    init_time = time.time()
    for i in clean:
        result_w2vneg300 = word2vec(i, 0, 5, 300)
        hf.create_dataset("w2vneg300" + i, data=result_w2vneg300)
    end_time = time.time()
    get_time_diff(init_time, end_time, "FE-Word2Vec NEG Vec200")

    # glove
    # for i in clean:
    #     result_fasttext = fasttext(i)
    return 1


def output_file():

    pass


def get_time_diff(init_time, end_time, desc):
    time_diff = end_time - init_time
    hours = int(time_diff // 3600)
    minutes = int((time_diff - (hours * 3600)) // 60)
    seconds = round(time_diff - (hours * 3600) - (minutes * 60), 3)
    with open(NEW_DIR + "/log.csv", "a") as csv:
        csv.write("{},{},{},{},{}\n".format(desc, hours, minutes, seconds, time_diff))
    csv.close()
    print(
        "Elapsed time for {}= {} hours, {} minutes, {} seconds".format(
            desc, hours, minutes, seconds
        )
    )


if __name__ == "__main__":
    print("Program started...")
    init_time = time.time()

    print("Doing preparation", end="\r")
    DATA_TEMPLATED = ["./Dataset/templated/"+x for x in os.listdir("./Dataset/templated")]
    DATA_CLEAN = ["./Dataset/clean/"+x for x in os.listdir("./Dataset/clean")]
    DATA_RAW = ["./Dataset/raw/"+x for x in os.listdir("./Dataset/raw")]
    with open(NEW_DIR + "/log.csv", "w") as csv:
        csv.write("Description,hours,minutes,seconds,total\n")
    csv.close()
    hf_name = "data" + TIMESTAMP + ".h5"
    hf = h5py.File(NEW_DIR + "/" + hf_name, "w")
    print("Preparation complete...")


    print("Running Feature Extraction")
    main(hf, DATA_TEMPLATED, DATA_CLEAN, DATA_RAW)
    hf.close()
    print("Feature Extraction Complete")

    end_time = time.time()
    get_time_diff(init_time, end_time, "FULL RUN")
