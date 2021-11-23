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

import featureSelection as featSel

# ; = template
# ;;; = templated

warnings.filterwarnings("ignore")
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
TARGET_DIR = "featureExtractionResults\\run-" + TIMESTAMP
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
NEW_DIR = os.path.join(CURRENT_DIR, TARGET_DIR)
os.mkdir(NEW_DIR)
os.mkdir(os.path.join(NEW_DIR, "models"))


def main(hf, templated, clean, raw):
    # Bigram
    init_time = time.time()
    for i in templated:
        print("Running: {:<20} --> {:<100}".format("Bigram", i), end="\r")
        result_bigram, feat_name = ngram(i, 2)
        hf.create_dataset("bigram" + i, data=result_bigram)
    end_time = time.time()
    get_time_diff(init_time, end_time, "FE-Bigram")

    # Trigram
    init_time = time.time()
    for i in templated:
        print("Running: {:<20} --> {:<100}".format("Trigram", i), end="\r")
        result_trigram, feat_name = ngram(i, 3)
        hf.create_dataset("trigram" + i, data=result_trigram)
    end_time = time.time()
    get_time_diff(init_time, end_time, "FE-trigram")

    # TF
    init_time = time.time()
    for i in templated:
        print("Running: {:<20} --> {:<100}".format("TF", i), end="\r")
        result_tf = term_freq(i, 1)
        hf.create_dataset("term_freq" + i, data=result_tf)
    end_time = time.time()
    get_time_diff(init_time, end_time, "FE-TF")

    # TF_IDF
    init_time = time.time()
    for i in templated:
        print("Running: {:<20} --> {:<100}".format("TFIDF", i), end="\r")
        result_tf_idf = run_tfidf_tweet(i)
        hf.create_dataset("tf_idf" + i, data=result_tf_idf)
    end_time = time.time()
    get_time_diff(init_time, end_time, "FE-TFIDF")

    # Orthography
    init_time = time.time()
    for i in templated:
        print("Running: {:<20} --> {:<100}".format("Orthography", i), end="\r")
        result_ortho, features = run_ortografi(i)
        hf.create_dataset("ortho" + i, data=result_ortho)
    end_time = time.time()
    get_time_diff(init_time, end_time, "FE-Orthography")

    # POS-tag
    init_time = time.time()
    for i in clean:
        print("Running: {:<20} --> {:<100}".format("POSTag", i), end="\r")
        result_pos = run_postag(i)
        hf.create_dataset("postag" + i, data=result_pos)
    end_time = time.time()
    get_time_diff(init_time, end_time, "FE-POS Tag")

    # Lex-Vania
    init_time = time.time()
    for i in clean:
        print("Running: {:<20} --> {:<100}".format("lex_van", i), end="\r")
        result_lexiconVania, sentimen = run_lexiconVania_tweet(i)
        hf.create_dataset("lex_vania" + i, data=result_lexiconVania)
    end_time = time.time()
    get_time_diff(init_time, end_time, "FE-Lexicon Vania")

    # Lex-InSet
    init_time = time.time()
    for i in clean:
        print("Running: {:<20} --> {:<100}".format("lex_ins", i), end="\r")
        result_lexiconInSet, sentimen = run_lexiconInset_tweet(i)
        hf.create_dataset("lex_inset" + i, data=result_lexiconInSet)
    end_time = time.time()
    get_time_diff(init_time, end_time, "FE-Lexicon InSet")

    # Lex-Combine
    init_time = time.time()
    for i in clean:
        print("Running: {:<20} --> {:<100}".format("lex_combined", i), end="\r")
        run_lexiconCombined, sentimen = run_lexiconCombined_tweet(i)
        hf.create_dataset("lex_combined" + i, data=run_lexiconCombined)
    end_time = time.time()
    get_time_diff(init_time, end_time, "FE-Lexicon Combined")


    # FastText CBOW
    init_time = time.time()
    for i in clean:
        print("Running: {:<20} --> {:<100}".format("FastText", i), end="\r")
        fasttext_current_dir = os.path.join(NEW_DIR, "models\\fasttext_{}".format(i.split("/")[-1]))
        os.mkdir(fasttext_current_dir)
        model_fasttext_cbow, result_fasttext_cbow = fasttext(i)
        model_fasttext_cbow.save(os.path.join(fasttext_current_dir, "fasttext_skipgram_model.model"))
        hf.create_dataset("fasttext_cbow_" + i, data=result_fasttext_cbow)
    end_time = time.time()
    get_time_diff(init_time, end_time, "FE-FastTextCBOW")

    # FastText SG
    init_time = time.time()
    for i in clean:
        print("Running: {:<20} --> {:<100}".format("FastTextSG", i), end="\r")
        fasttext_current_dir = os.path.join(NEW_DIR, "models\\fasttextSG_{}".format(i.split("/")[-1]))
        os.mkdir(fasttext_current_dir)
        model_fasttext_sg, result_fasttext_sg = fasttextsg(i)
        model_fasttext_sg.save(os.path.join(fasttext_current_dir, "fasttext_skipgram_model.model"))
        hf.create_dataset("fasttext_sg_" + i, data=result_fasttext_sg)
    end_time = time.time()
    get_time_diff(init_time, end_time, "FE-FastTextSG")


    # WORD2VEC 100
    init_time = time.time()
    for i in clean:
        print("Running: {:<20} --> {:<100}".format("Word2Vec CBHS100", i), end="\r")
        w2v_current_dir = os.path.join(NEW_DIR, "models\\W2VCBHS100_{}".format(i.split("/")[-1]))
        os.mkdir(w2v_current_dir)
        model_w2vhs100, feat_w2vhs100 = word2vec_cbow(i, 1, 0, 100)
        model_w2vhs100.save(os.path.join(w2v_current_dir, "word2vec_cbow_model.model"))
        hf.create_dataset("w2v_cbhs_100" + i, data=feat_w2vhs100)
    end_time = time.time()
    get_time_diff(init_time, end_time, "FE-Word2Vec CBHS Vec100")

    init_time = time.time()
    for i in clean:
        print("Running: {:<20} --> {:<100}".format("Word2Vec CBNEG100", i), end="\r")
        w2v_current_dir = os.path.join(NEW_DIR, "models\\W2VCBNEG100_{}".format(i.split("/")[-1]))
        os.mkdir(w2v_current_dir)
        model_w2vneg100, result_w2vneg100 = word2vec_cbow(i, 0, 5, 100)
        model_w2vneg100.save(os.path.join(w2v_current_dir, "word2vec_cbow_model.model"))
        hf.create_dataset("w2v_cbneg_100" + i, data=result_w2vneg100)
    end_time = time.time()
    get_time_diff(init_time, end_time, "FE-Word2Vec CBNEG Vec100")
    
    init_time = time.time()
    for i in clean:
        print("Running: {:<20} --> {:<100}".format("Word2Vec SGHS100", i), end="\r")
        w2v_current_dir = os.path.join(NEW_DIR, "models\\W2VSGHS100_{}".format(i.split("/")[-1]))
        os.mkdir(w2v_current_dir)
        model_w2vhs100, result_w2vhs100 = word2vec_sg(i, 1, 0, 100)
        model_w2vhs100.save(os.path.join(w2v_current_dir, "word2vec_skipgram_model.model"))
        hf.create_dataset("w2v_sghs_100" + i, data=result_w2vhs100)
    end_time = time.time()
    get_time_diff(init_time, end_time, "FE-Word2Vec SGHS Vec100")

    init_time = time.time()
    for i in clean:
        print("Running: {:<20} --> {:<100}".format("Word2Vec SGNEG100", i), end="\r")
        w2v_current_dir = os.path.join(NEW_DIR, "models\\W2VSGNEG100_{}".format(i.split("/")[-1]))
        os.mkdir(w2v_current_dir)
        model_w2vneg100, result_w2vneg100 = word2vec_sg(i, 0, 5, 100)
        model_w2vneg100.save(os.path.join(w2v_current_dir, "word2vec_skipgram_model.model"))
        hf.create_dataset("w2v_sgneg_100" + i, data=result_w2vneg100)
    end_time = time.time()
    get_time_diff(init_time, end_time, "FE-Word2Vec SGNEG Vec100")

    # WORD2VEC 200
    init_time = time.time()
    for i in clean:
        print("Running: {:<20} --> {:<100}".format("Word2Vec CBHS200", i), end="\r")
        w2v_current_dir = os.path.join(NEW_DIR, "models\\W2VCBHS200_{}".format(i.split("/")[-1]))
        os.mkdir(w2v_current_dir)
        model_w2vhs200, result_w2vhs200 = word2vec_cbow(i, 1, 0, 200)
        model_w2vhs200.save(os.path.join(w2v_current_dir, "word2vec_cbow_model.model"))
        hf.create_dataset("w2v_cbhs_200" + i, data=result_w2vhs200)
    end_time = time.time()
    get_time_diff(init_time, end_time, "FE-Word2Vec CBHS Vec200")

    init_time = time.time()
    for i in clean:
        print("Running: {:<20} --> {:<100}".format("Word2Vec CBNEG200", i), end="\r")
        w2v_current_dir = os.path.join(NEW_DIR, "models\\W2VCBNEG200_{}".format(i.split("/")[-1]))
        os.mkdir(w2v_current_dir)
        model_w2vneg200, result_w2vneg200 = word2vec_cbow(i, 0, 5, 200)
        model_w2vneg200.save(os.path.join(w2v_current_dir, "word2vec_cbow_model.model"))
        hf.create_dataset("w2v_cbneg_200" + i, data=result_w2vneg200)
    end_time = time.time()
    get_time_diff(init_time, end_time, "FE-Word2Vec CBNEG Vec200")
    
    init_time = time.time()
    for i in clean:
        print("Running: {:<20} --> {:<100}".format("Word2Vec SGHS200", i), end="\r")
        w2v_current_dir = os.path.join(NEW_DIR, "models\\W2VSGHS200_{}".format(i.split("/")[-1]))
        os.mkdir(w2v_current_dir)
        model_w2vhs200, result_w2vhs200 = word2vec_sg(i, 1, 0, 200)
        model_w2vhs200.save(os.path.join(w2v_current_dir, "word2vec_skipgram_model.model"))
        hf.create_dataset("w2v_sghs_200" + i, data=result_w2vhs200)
    end_time = time.time()
    get_time_diff(init_time, end_time, "FE-Word2Vec SGHS Vec200")

    init_time = time.time()
    for i in clean:
        print("Running: {:<20} --> {:<100}".format("Word2Vec SGNEG200", i), end="\r")
        w2v_current_dir = os.path.join(NEW_DIR, "models\\W2VSGNEG200_{}".format(i.split("/")[-1]))
        os.mkdir(w2v_current_dir)
        model_w2vneg200, result_w2vneg200 = word2vec_sg(i, 0, 5, 200)
        model_w2vneg200.save(os.path.join(w2v_current_dir, "word2vec_skipgram_model.model"))
        hf.create_dataset("w2v_sgneg_200" + i, data=result_w2vneg200)
    end_time = time.time()
    get_time_diff(init_time, end_time, "FE-Word2Vec SGNEG Vec200")

    # WORD2VEC 300
    init_time = time.time()
    for i in clean:
        print("Running: {:<20} --> {:<100}".format("Word2Vec CBHS300", i), end="\r")
        w2v_current_dir = os.path.join(NEW_DIR, "models\\W2VCBHS300_{}".format(i.split("/")[-1]))
        os.mkdir(w2v_current_dir)
        model_w2vhs300, result_w2vhs300 = word2vec_cbow(i, 1, 0, 300)
        model_w2vhs300.save(os.path.join(w2v_current_dir, "word2vec_cbow_model.model"))
        hf.create_dataset("w2v_cbhs_300" + i, data=result_w2vhs300)
    end_time = time.time()
    get_time_diff(init_time, end_time, "FE-Word2Vec CBHS Vec300")

    init_time = time.time()
    for i in clean:
        print("Running: {:<20} --> {:<100}".format("Word2Vec CBNEG300", i), end="\r")
        w2v_current_dir = os.path.join(NEW_DIR, "models\\W2VCBNEG300_{}".format(i.split("/")[-1]))
        os.mkdir(w2v_current_dir)
        model_w2vneg300, result_w2vneg300 = word2vec_cbow(i, 0, 5, 300)
        model_w2vneg300.save(os.path.join(w2v_current_dir, "word2vec_cbow_model.model"))
        hf.create_dataset("w2v_cbneg_300" + i, data=result_w2vneg300)
    end_time = time.time()
    get_time_diff(init_time, end_time, "FE-Word2Vec CBNEG Vec300")
    
    init_time = time.time()
    for i in clean:
        print("Running: {:<20} --> {:<100}".format("Word2Vec SGHS300", i), end="\r")
        w2v_current_dir = os.path.join(NEW_DIR, "models\\W2VSGHS300_{}".format(i.split("/")[-1]))
        os.mkdir(w2v_current_dir)
        model_w2vhs300, result_w2vhs300 = word2vec_sg(i, 1, 0, 300)
        model_w2vhs300.save(os.path.join(w2v_current_dir, "word2vec_skipgram_model.model"))
        hf.create_dataset("w2v_sghs_300" + i, data=result_w2vhs300)
    end_time = time.time()
    get_time_diff(init_time, end_time, "FE-Word2Vec SGHS Vec300")

    init_time = time.time()
    for i in clean:
        print("Running: {:<20} --> {:<100}".format("Word2Vec SGNEG300", i), end="\r")
        w2v_current_dir = os.path.join(NEW_DIR, "models\\W2VSGNEG300_{}".format(i.split("/")[-1]))
        os.mkdir(w2v_current_dir)
        model_w2vneg300, result_w2vneg300 = word2vec_sg(i, 0, 5, 300)
        model_w2vneg300.save(os.path.join(w2v_current_dir, "word2vec_skipgram_model.model"))
        hf.create_dataset("w2v_sgneg_300" + i, data=result_w2vneg300)
    end_time = time.time()
    get_time_diff(init_time, end_time, "FE-Word2Vec SGNEG Vec300")


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


if __name__ == "__main__":
    print("Program started with output on {}".format(TARGET_DIR))
    init_time = time.time()

    print("Doing preparation", end="\r")
    DATA_TEMPLATED = ["./Dataset/templated/" + x for x in os.listdir("./Dataset/templated")]
    DATA_CLEAN = ["./Dataset/clean/" + x for x in os.listdir("./Dataset/clean")]
    DATA_RAW = ["./Dataset/raw/" + x for x in os.listdir("./Dataset/raw")]

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
