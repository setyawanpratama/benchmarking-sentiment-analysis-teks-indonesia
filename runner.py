import os
import h5py
from nltk.util import usage
import numpy as np
import datetime
import time
import warnings

from featureExtraction.tf.tf import *
from featureExtraction.tfidf.tfidf import *
from featureExtraction.fasttext.fasttext import *
from featureExtraction.lexicon.lexicon import *
from featureExtraction.orthography.orthography import *
from featureExtraction.pos.postag import *
from featureExtraction.tf.tf import *
from featureExtraction.word2vec.word2vec import *
from featureExtraction.ngram.ngram import *

from featureSelection.kFold import *

# ; = template
# ;;; = templated

warnings.filterwarnings("ignore")
TARGET_DIR = "featureExtractionResults\\run-" + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
NEW_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), TARGET_DIR)
os.mkdir(NEW_DIR)
os.mkdir(os.path.join(NEW_DIR, "models"))

DATA_TEMPLATED = ["./Dataset/templated/" + x for x in os.listdir("./Dataset/templated")]
DATA_TEMPLATED.sort()
DATA_CLEAN = ["./Dataset/clean/" + x for x in os.listdir("./Dataset/clean")]
DATA_CLEAN.sort()

ALL_FE_DATASET = []

print("Preparation Completed")

def print_info(item, dataset):
    print("Running: {:<20} --> {:<100}".format(item, dataset), end="\r")

# TF
def fe_term_frequency(hf):
    init_time = time.time()
    for i in DATA_TEMPLATED:
        print_info("TF", i)
        result_tf = term_freq(i, 1)
        hf.create_dataset("tf_" + DATA_TEMPLATED.index(i), data=result_tf)
    get_time_diff(init_time, time.time(), "FE-TF")

# Bigram
def fe_bigram(hf):
    init_time = time.time()
    for i in DATA_TEMPLATED:
        print("Running: {:<20} --> {:<100}".format("Bigram", i), end="\r")
        result_bigram, feat_name = ngram(i, 2)
        hf.create_dataset("bigram_" + DATA_TEMPLATED.index(i), data=result_bigram)
    get_time_diff(init_time, time.time(), "FE-Bigram")

# Trigram
def fe_trigram(hf):
    init_time = time.time()
    for i in DATA_TEMPLATED:
        print("Running: {:<20} --> {:<100}".format("Trigram", i), end="\r")
        result_trigram, feat_name = ngram(i, 3)
        hf.create_dataset("trigram_" + DATA_TEMPLATED.index(i), data=result_trigram)
    get_time_diff(init_time, time.time(), "FE-trigram")

# TF_IDF
def fe_tfidf(hf):
    init_time = time.time()
    for i in DATA_TEMPLATED:
        print("Running: {:<20} --> {:<100}".format("TFIDF", i), end="\r")
        result_tf_idf = run_tfidf_tweet(i)
        hf.create_dataset("tfidf_" + DATA_TEMPLATED.index(i), data=result_tf_idf)
    get_time_diff(init_time, time.time(), "FE-TFIDF")

# Orthography
def fe_orthography(hf):
    init_time = time.time()
    for i in DATA_TEMPLATED:
        print("Running: {:<20} --> {:<100}".format("Orthography", i), end="\r")
        result_ortho, features = run_ortografi(i)
        hf.create_dataset("ortho_" + DATA_TEMPLATED.index(i), data=result_ortho)
    get_time_diff(init_time, time.time(), "FE-Orthography")

# POS-tag
def fe_postag(hf):
    init_time = time.time()
    for i in DATA_CLEAN:
        print("Running: {:<20} --> {:<100}".format("POSTag", i), end="\r")
        result_pos = run_postag(i)
        hf.create_dataset("postag_" + DATA_CLEAN.index(i), data=result_pos)
    get_time_diff(init_time, time.time(), "FE-POS Tag")

# Lex-Vania
def fe_lexicon_vania(hf):
    init_time = time.time()
    for i in DATA_CLEAN:
        print("Running: {:<20} --> {:<100}".format("lex_van", i), end="\r")
        result_lexiconVania, sentimen = run_lexiconVania_tweet(i)
        hf.create_dataset("lex_vania_" + DATA_CLEAN.index(i), data=result_lexiconVania)
    get_time_diff(init_time, time.time(), "FE-Lexicon Vania")

# Lex-InSet
def fe_lexicon_inset(hf):
    init_time = time.time()
    for i in DATA_CLEAN:
        print("Running: {:<20} --> {:<100}".format("lex_ins", i), end="\r")
        result_lexiconInSet, sentimen = run_lexiconInset_tweet(i)
        hf.create_dataset("lex_inset_" + DATA_CLEAN.index(i), data=result_lexiconInSet)
    get_time_diff(init_time, time.time(), "FE-Lexicon InSet")

def fe_lexicon_combined(hf):
    # Lex-Combine
    init_time = time.time()
    for i in DATA_CLEAN:
        print("Running: {:<20} --> {:<100}".format("lex_combined", i), end="\r")
        run_lexiconCombined, sentimen = run_lexiconCombined_tweet(i)
        hf.create_dataset("lex_combined_" + DATA_CLEAN.index(i), data=run_lexiconCombined)
    get_time_diff(init_time, time.time(), "FE-Lexicon Combined")

# FastText CBOW 100
def fe_fasttext_cbow_100(hf, save_models):
    init_time = time.time()
    for i in DATA_CLEAN:
        print("Running: {:<20} --> {:<100}".format("FastText 100", i), end="\r")
        model_ft_cb_100, result_ft_cb_100 = fasttext(i, 100)
        hf.create_dataset("fasttext_cb_100_" + DATA_CLEAN.index(i), data=result_ft_cb_100)
        if save_models:
            fasttext_current_dir = os.path.join(NEW_DIR, "models\\fasttext100_{}".format(i.split("/")[-1]))
            os.mkdir(fasttext_current_dir)
            model_ft_cb_100.save(os.path.join(fasttext_current_dir, "fasttext_skipgram_model_100.model"))
    get_time_diff(init_time, time.time(), "FE-FastTextCBOW Vec100")

# FastText CBOW 200
def fe_fasttext_cbow_200(hf, save_models):
    init_time = time.time()
    for i in DATA_CLEAN:
        print("Running: {:<20} --> {:<100}".format("FastText 200", i), end="\r")
        model_ft_cb_200, result_ft_cb_200 = fasttext(i, 200)
        if save_models:
            fasttext_current_dir = os.path.join(NEW_DIR, "models\\fasttext200_{}".format(i.split("/")[-1]))
            os.mkdir(fasttext_current_dir)
            model_ft_cb_200.save(os.path.join(fasttext_current_dir, "fasttext_skipgram_model_200.model"))
        hf.create_dataset("fasttext_cb_200_" + DATA_CLEAN.index(i), data=result_ft_cb_200)
    get_time_diff(init_time, time.time(), "FE-FastTextCBOW Vec200")

# FastText CBOW 300
def fe_fasttext_cbow_300(hf, save_models):
    init_time = time.time()
    for i in DATA_CLEAN:
        print("Running: {:<20} --> {:<100}".format("FastText 300", i), end="\r")
        model_ft_cb_300, result_ft_cb_300 = fasttext(i, 300)
        if save_models:
            fasttext_current_dir = os.path.join(NEW_DIR, "models\\fasttext300_{}".format(i.split("/")[-1]))
            os.mkdir(fasttext_current_dir)
            model_ft_cb_300.save(os.path.join(fasttext_current_dir, "fasttext_skipgram_model_300.model"))
        hf.create_dataset("fasttext_cb_300_" + DATA_CLEAN.index(i), data=result_ft_cb_300)
    get_time_diff(init_time, time.time(), "FE-FastTextCBOW Vec300")

# FastText SG 100
def fe_fasttext_sg_100(hf, save_models):
    init_time = time.time()
    for i in DATA_CLEAN:
        print("Running: {:<20} --> {:<100}".format("FastTextSG 100", i), end="\r")
        model_ft_sg_100, result_ft_sg_100 = fasttextsg(i, 100)
        if save_models:
            fasttext_current_dir = os.path.join(NEW_DIR, "models\\fasttextSG100_{}".format(i.split("/")[-1]))
            os.mkdir(fasttext_current_dir)
            model_ft_sg_100.save(os.path.join(fasttext_current_dir, "fasttext_skipgram_model_100.model"))
        hf.create_dataset("fasttext_sg_100_" + DATA_CLEAN.index(i), data=result_ft_sg_100)
    get_time_diff(init_time, time.time(), "FE-FastTextSG Vec100")    

# FastText SG 200
def fe_fasttext_sg_200(hf, save_models):
    init_time = time.time()
    for i in DATA_CLEAN:
        print("Running: {:<20} --> {:<100}".format("FastTextSG 200", i), end="\r")
        model_ft_sg_200, result_ft_sg_200 = fasttextsg(i, 200)
        if save_models:
            fasttext_current_dir = os.path.join(NEW_DIR, "models\\fasttextSG200_{}".format(i.split("/")[-1]))
            os.mkdir(fasttext_current_dir)
            model_ft_sg_200.save(os.path.join(fasttext_current_dir, "fasttext_skipgram_model_200.model"))
        hf.create_dataset("fasttext_sg_200_" + DATA_CLEAN.index(i), data=result_ft_sg_200)
    get_time_diff(init_time, time.time(), "FE-FastTextSG Vec200")

# FastText SG 300
def fe_fasttext_sg_300(hf, save_models):
    init_time = time.time()
    for i in DATA_CLEAN:
        print("Running: {:<20} --> {:<100}".format("FastTextSG 300", i), end="\r")
        model_ft_sg_300, result_ft_sg_300 = fasttextsg(i, 300)
        if save_models:
            fasttext_current_dir = os.path.join(NEW_DIR, "models\\fasttextSG300_{}".format(i.split("/")[-1]))
            os.mkdir(fasttext_current_dir)
            model_ft_sg_300.save(os.path.join(fasttext_current_dir, "fasttext_skipgram_model_300.model"))
        hf.create_dataset("fasttext_sg_300_" + DATA_CLEAN.index(i), data=result_ft_sg_300)
    get_time_diff(init_time, time.time(), "FE-FastTextSG Vec300")

# WORD2VEC 100
def fe_word2vec_cbow_hs_100(hf, save_models):
    init_time = time.time()
    for i in DATA_CLEAN:
        print("Running: {:<20} --> {:<100}".format("Word2Vec CBHS100", i), end="\r")
        model_w2vcbhs100, feat_w2vcbhs100 = word2vec_cbow(i, 1, 0, 100)
        if save_models:
            w2v_current_dir = os.path.join(NEW_DIR, "models\\W2VCBHS100_{}".format(i.split("/")[-1]))
            os.mkdir(w2v_current_dir)
            model_w2vcbhs100.save(os.path.join(w2v_current_dir, "word2vec_cbow_model.model"))
        hf.create_dataset("w2v_cbow_hs_100_" + DATA_CLEAN.index(i), data=feat_w2vcbhs100)
    get_time_diff(init_time, time.time(), "FE-Word2Vec CBHS Vec100")
    
def fe_word2vec_cbow_neg_100(hf, save_models):
    init_time = time.time()
    for i in DATA_CLEAN:
        print("Running: {:<20} --> {:<100}".format("Word2Vec CBNEG100", i), end="\r")
        model_w2vneg100, result_w2vneg100 = word2vec_cbow(i, 0, 5, 100)
        if save_models:
            w2v_current_dir = os.path.join(NEW_DIR, "models\\W2VCBNEG100_{}".format(i.split("/")[-1]))
            os.mkdir(w2v_current_dir)
            model_w2vneg100.save(os.path.join(w2v_current_dir, "word2vec_cbow_model.model"))
        hf.create_dataset("w2v_cbow_neg_100" + DATA_CLEAN.index(i), data=result_w2vneg100)
    end_time = time.time()
    get_time_diff(init_time, end_time, "FE-Word2Vec CBNEG Vec100")

def fe_word2vec_sg_hs_100(hf, save_models):
    init_time = time.time()
    for i in DATA_CLEAN:
        print("Running: {:<20} --> {:<100}".format("Word2Vec SGHS100", i), end="\r")
        model_w2vhs100, result_w2vhs100 = word2vec_sg(i, 1, 0, 100)
        if save_models:
            w2v_current_dir = os.path.join(NEW_DIR, "models\\W2VSGHS100_{}".format(i.split("/")[-1]))
            os.mkdir(w2v_current_dir)
            model_w2vhs100.save(os.path.join(w2v_current_dir, "word2vec_skipgram_model.model"))
        hf.create_dataset("w2v_sg_hs_100_" + DATA_CLEAN.index(i), data=result_w2vhs100)
    get_time_diff(init_time, time.time(), "FE-Word2Vec SGHS Vec100")

def fe_word2vec_sg_neg_100(hf, save_models):
    init_time = time.time()
    for i in DATA_CLEAN:
        print("Running: {:<20} --> {:<100}".format("Word2Vec SGNEG100", i), end="\r")
        model_w2vneg100, result_w2vneg100 = word2vec_sg(i, 0, 5, 100)
        if save_models:
            w2v_current_dir = os.path.join(NEW_DIR, "models\\W2VSGNEG100_{}".format(i.split("/")[-1]))
            os.mkdir(w2v_current_dir)
            model_w2vneg100.save(os.path.join(w2v_current_dir, "word2vec_skipgram_model.model"))
        hf.create_dataset("w2v_sg_neg_100_" + DATA_CLEAN.index(i), data=result_w2vneg100)
    get_time_diff(init_time, time.time(), "FE-Word2Vec SGNEG Vec100")

# WORD2VEC 200
def fe_word2vec_cbow_hs_200(hf, save_models):
    init_time = time.time()
    for i in DATA_CLEAN:
        print("Running: {:<20} --> {:<100}".format("Word2Vec CBHS200", i), end="\r")
        model_w2vhs200, result_w2vhs200 = word2vec_cbow(i, 1, 0, 200)
        if save_models:
            w2v_current_dir = os.path.join(NEW_DIR, "models\\W2VCBHS200_{}".format(i.split("/")[-1]))
            os.mkdir(w2v_current_dir)
            model_w2vhs200.save(os.path.join(w2v_current_dir, "word2vec_cbow_model.model"))
        hf.create_dataset("w2v_cbow_hs_200_" + DATA_CLEAN.index(i), data=result_w2vhs200)
    get_time_diff(init_time, time.time(), "FE-Word2Vec CBHS Vec200")

def fe_word2vec_cbow_neg_200(hf, save_models):
    init_time = time.time()
    for i in DATA_CLEAN:
        print("Running: {:<20} --> {:<100}".format("Word2Vec CBNEG200", i), end="\r")
        model_w2vneg200, result_w2vneg200 = word2vec_cbow(i, 0, 5, 200)
        if save_models:
            w2v_current_dir = os.path.join(NEW_DIR, "models\\W2VCBNEG200_{}".format(i.split("/")[-1]))
            os.mkdir(w2v_current_dir)
            model_w2vneg200.save(os.path.join(w2v_current_dir, "word2vec_cbow_model.model"))
        hf.create_dataset("w2v_cbow_neg_200_" + DATA_CLEAN.index(i), data=result_w2vneg200)
    end_time = time.time()
    get_time_diff(init_time, end_time, "FE-Word2Vec CBNEG Vec200")

def fe_word2vec_sg_hs_200(hf, save_models):
    init_time = time.time()
    for i in DATA_CLEAN:
        print("Running: {:<20} --> {:<100}".format("Word2Vec SGHS200", i), end="\r")
        model_w2vhs200, result_w2vhs200 = word2vec_sg(i, 1, 0, 200)
        if save_models:
            w2v_current_dir = os.path.join(NEW_DIR, "models\\W2VSGHS200_{}".format(i.split("/")[-1]))
            os.mkdir(w2v_current_dir)
            model_w2vhs200.save(os.path.join(w2v_current_dir, "word2vec_skipgram_model.model"))
        hf.create_dataset("w2v_sg_hs_200_" + DATA_CLEAN.index(i), data=result_w2vhs200)
    end_time = time.time()
    get_time_diff(init_time, end_time, "FE-Word2Vec SGHS Vec200")

def fe_word2vec_sg_neg_200(hf, save_models):
    init_time = time.time()
    for i in DATA_CLEAN:
        print("Running: {:<20} --> {:<100}".format("Word2Vec SGNEG200", i), end="\r")
        model_w2vneg200, result_w2vneg200 = word2vec_sg(i, 0, 5, 200)
        if save_models:
            w2v_current_dir = os.path.join(NEW_DIR, "models\\W2VSGNEG200_{}".format(i.split("/")[-1]))
            os.mkdir(w2v_current_dir)
            model_w2vneg200.save(os.path.join(w2v_current_dir, "word2vec_skipgram_model.model"))
        hf.create_dataset("w2v_sg_neg_200_" + DATA_CLEAN.index(i), data=result_w2vneg200)
    end_time = time.time()
    get_time_diff(init_time, end_time, "FE-Word2Vec SGNEG Vec200")

# WORD2VEC 300
def fe_word2vec_cbow_hs_300(hf, save_models):
    init_time = time.time()
    for i in DATA_CLEAN:
        print("Running: {:<20} --> {:<100}".format("Word2Vec CBHS300", i), end="\r")
        model_w2vhs300, result_w2vhs300 = word2vec_cbow(i, 1, 0, 300)
        if save_models:
            w2v_current_dir = os.path.join(NEW_DIR, "models\\W2VCBHS300_{}".format(i.split("/")[-1]))
            os.mkdir(w2v_current_dir)
            model_w2vhs300.save(os.path.join(w2v_current_dir, "word2vec_cbow_model.model"))
        hf.create_dataset("w2v_cbow_hs_300_" + DATA_CLEAN.index(i), data=result_w2vhs300)
    end_time = time.time()
    get_time_diff(init_time, end_time, "FE-Word2Vec CBHS Vec300")

def fe_word2vec_cbow_neg_300(hf, save_models):
    init_time = time.time()
    for i in DATA_CLEAN:
        print("Running: {:<20} --> {:<100}".format("Word2Vec CBNEG300", i), end="\r")
        model_w2vneg300, result_w2vneg300 = word2vec_cbow(i, 0, 5, 300)
        if save_models:
            w2v_current_dir = os.path.join(NEW_DIR, "models\\W2VCBNEG300_{}".format(i.split("/")[-1]))
            os.mkdir(w2v_current_dir)
            model_w2vneg300.save(os.path.join(w2v_current_dir, "word2vec_cbow_model.model"))
        hf.create_dataset("w2v_cbow_neg_300_" + DATA_CLEAN.index(i), data=result_w2vneg300)
    end_time = time.time()
    get_time_diff(init_time, end_time, "FE-Word2Vec CBNEG Vec300")

def fe_word2vec_sg_hs_300(hf, save_models):    
    init_time = time.time()
    for i in DATA_CLEAN:
        print("Running: {:<20} --> {:<100}".format("Word2Vec SGHS300", i), end="\r")
        model_w2vhs300, result_w2vhs300 = word2vec_sg(i, 1, 0, 300)
        if save_models:
            w2v_current_dir = os.path.join(NEW_DIR, "models\\W2VSGHS300_{}".format(i.split("/")[-1]))
            os.mkdir(w2v_current_dir)
            model_w2vhs300.save(os.path.join(w2v_current_dir, "word2vec_skipgram_model.model"))
        hf.create_dataset("w2v_sg_hs_300_" + DATA_CLEAN.index(i), data=result_w2vhs300)
    end_time = time.time()
    get_time_diff(init_time, end_time, "FE-Word2Vec SGHS Vec300")

def fe_word2vec_sg_neg_300(hf, save_models):
    init_time = time.time()
    for i in DATA_CLEAN:
        print("Running: {:<20} --> {:<100}".format("Word2Vec SGNEG300", i), end="\r")
        model_w2vneg300, result_w2vneg300 = word2vec_sg(i, 0, 5, 300)
        if save_models:
            w2v_current_dir = os.path.join(NEW_DIR, "models\\W2VSGNEG300_{}".format(i.split("/")[-1]))
            os.mkdir(w2v_current_dir)
            model_w2vneg300.save(os.path.join(w2v_current_dir, "word2vec_skipgram_model.model"))
        hf.create_dataset("w2v_sg_neg_300_" + DATA_CLEAN.index(i), data=result_w2vneg300)
    end_time = time.time()
    get_time_diff(init_time, end_time, "FE-Word2Vec SGNEG Vec300")

def run_logistic_regression(hf):
    kFold()


def get_time_diff(init_time, end_time, desc):
    time_diff = end_time - init_time
    hours = int(time_diff // 3600)
    minutes = int((time_diff - (hours * 3600)) // 60)
    seconds = round(time_diff - (hours * 3600) - (minutes * 60), 3)

    with open(NEW_DIR + "/log.csv", "a") as csv:
        csv.write("{},{},{},{},{}\n".format(desc, hours, minutes, seconds, time_diff))
    csv.close()

    status = "Elapsed time for {:<30} = {} hours, {} minutes, {} seconds".format(
        desc, hours, minutes, seconds
    )
    print("{:<150}".format(status))

def main():
    print("Program started with output on {}".format(TARGET_DIR))
    init_time = time.time()
    
    with open(NEW_DIR + "/dataset.csv", "w") as csv:
        csv.write("index,data_clean,data_templated\n")
        for i in range(max(len(DATA_CLEAN)-1, len(DATA_TEMPLATED)-1)):
            temp_clean = DATA_CLEAN[i] if i <= len(DATA_CLEAN)-1 else ''
            temp_templ = DATA_TEMPLATED[i] if i <= len(DATA_TEMPLATED)-1 else ''
            csv.write("{},{},{}".format(i, temp_clean, temp_templ))
    csv.close()
    
    with open(NEW_DIR + "/log.csv", "w") as csv:
        csv.write("Description,hours,minutes,seconds,total\n")
    csv.close()
    hf_name = "data.h5"
    hf = h5py.File(NEW_DIR + "/" + hf_name, "w")
    print("Preparation complete...")

    print("Running Feature Extraction")
    # TODO: insert if here
    # Ngram
    fe_term_frequency(hf, templated=DATA_TEMPLATED)
    fe_bigram(hf, templated=DATA_TEMPLATED)
    fe_trigram(hf, templated=DATA_TEMPLATED)
    fe_tfidf(hf, templated=DATA_TEMPLATED)

    fe_orthography(hf, templated=DATA_TEMPLATED)

    fe_postag(hf, clean=DATA_CLEAN)

    # Lexicon
    fe_lexicon_vania(hf, clean=DATA_CLEAN)
    fe_lexicon_inset(hf, clean=DATA_CLEAN)
    fe_lexicon_combined(hf, clean=DATA_CLEAN)

    # FastText
    fe_fasttext_cbow_100(hf, clean=DATA_CLEAN, save_models=False)
    fe_fasttext_cbow_200(hf, clean=DATA_CLEAN, save_models=False)
    fe_fasttext_cbow_300(hf, clean=DATA_CLEAN, save_models=False)
    fe_fasttext_sg_100(hf, clean=DATA_CLEAN, save_models=False)
    fe_fasttext_sg_200(hf, clean=DATA_CLEAN, save_models=False)
    fe_fasttext_sg_300(hf, clean=DATA_CLEAN, save_models=False)

    # Word2Vec
    fe_word2vec_cbow_hs_100(hf, clean=DATA_CLEAN, save_models=False)
    fe_word2vec_cbow_hs_200(hf, clean=DATA_CLEAN, save_models=False)
    fe_word2vec_cbow_hs_300(hf, clean=DATA_CLEAN, save_models=False)

    fe_word2vec_cbow_neg_100(hf, clean=DATA_CLEAN, save_models=False)
    fe_word2vec_cbow_neg_200(hf, clean=DATA_CLEAN, save_models=False)
    fe_word2vec_cbow_neg_300(hf, clean=DATA_CLEAN, save_models=False)

    fe_word2vec_sg_hs_100(hf, clean=DATA_CLEAN, save_models=False)
    fe_word2vec_sg_hs_200(hf, clean=DATA_CLEAN, save_models=False)
    fe_word2vec_sg_hs_300(hf, clean=DATA_CLEAN, save_models=False)

    fe_word2vec_sg_neg_100(hf, clean=DATA_CLEAN, save_models=False)
    fe_word2vec_sg_neg_200(hf, clean=DATA_CLEAN, save_models=False)
    fe_word2vec_sg_neg_300(hf, clean=DATA_CLEAN, save_models=False)

    hf.close()
    print("Feature Extraction Complete")

    print("Starting classification... ")


    end_time = time.time()
    get_time_diff(init_time, end_time, "FULL RUN")

if __name__ == "__main__":
    main()
