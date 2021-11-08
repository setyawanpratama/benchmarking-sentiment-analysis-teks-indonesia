import nltk
import pandas as pd

def ekstraksi_ortografi(filename):    
    df = pd.read_csv(filename, delimiter=";;;", engine='python', header=0)

    all_orto_feat = []
    for tw in df["tweet"].tolist():
        exclamation_count = sum((1 for c in tw if c == "!"))
        word_len = len(nltk.word_tokenize(tw))
        char_len = len(tw)
        orto_feat = [exclamation_count, word_len, char_len]
        all_orto_feat.append(orto_feat)

    df_hasil = pd.DataFrame(all_orto_feat, columns=["exclamation_count", "word_len", "char_len"])
    return df_hasil