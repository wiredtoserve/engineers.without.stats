import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import nltk
import re
import string
import pandas as pd
import sklearn as sk
import pickle
import os

from ipdb import set_trace


def preprocess_string(sentence):

    # Remove ||| from kaggle dataset
    sentence = re.sub("[]|||[]", " ", sentence)

    # remove reddit subreddit urls
    sentence = re.sub("/r/[0-9A-Za-z]", "", sentence)

    # remove http
    sentence = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', sentence)

    # remove puncuation
    sentence = "".join([ci for ci in sentence if ci not in string.punctuation])

    # remove MBTI types
    MBTI_types = ['INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFJ', 'INFP', 'ENFP',
              'ISFP', 'ISTP', 'ISFJ', 'ISTJ', 'ESTP', 'ESFP', 'ESTJ', 'ESFJ',
              'MBTI']
    MBTI_types = [ti.lower() for ti in MBTI_types]

    tmp = []
    tokens = nlp(sentence)
    for token in tokens:
        if token.text.lower().strip() not in MBTI_types:
            tmp.append(token.text)

    # remove stop words
    tmp = [word for word in tmp if word not in STOP_WORDS]

    return " ".join(tmp)


def calc_NRC_vals(sentence):
    tokens = nlp(sentence)

    d = {}
    d["VAD_count"] = 0
    d["token_count"] = 0
    d["val"] = 0.0
    d["dom"] = 0.0
    d["arou"] = 0.0
    d["sent_pos"] = 0
    d["sent_neg"] = 0
    d["sent_ang"] = 0
    d["sent_ant"] = 0
    d["sent_dis"] = 0
    d["sent_fear"] = 0
    d["sent_joy"] = 0
    d["sent_sad"] = 0
    d["sent_surp"] = 0
    d["sent_trust"] = 0
    for token in tokens:
        t = token.lower_
        if t in NRC_dict:
            d["VAD_count"] += 1
            d["token_count"] += 1

            d["val"] += NRC_dict[t]["Valence"]
            d["dom"] += NRC_dict[t]["Dominance"]
            d["arou"] += NRC_dict[t]["Arousal"]

            d["sent_pos"] += NRC_dict[t]["Positive"]
            d["sent_neg"] += NRC_dict[t]["Negative"]
            d["sent_ang"] += NRC_dict[t]["Anger"]
            d["sent_ant"] += NRC_dict[t]["Anticipation"]
            d["sent_dis"] += NRC_dict[t]["Disgust"]
            d["sent_fear"] += NRC_dict[t]["Fear"]
            d["sent_joy"] += NRC_dict[t]["Joy"]
            d["sent_sad"] += NRC_dict[t]["Sadness"]
            d["sent_surp"] += NRC_dict[t]["Surprise"]
            d["sent_trust"] += NRC_dict[t]["Trust"]

    # compute fractions
    if d["VAD_count"] > 0:
        d["val"] = d["val"]/d["VAD_count"]
        d["dom"] = d["dom"]/d["VAD_count"]
        d["arou"] = d["arou"]/d["VAD_count"]

        d["sent_pos"] = d["sent_pos"]/d["token_count"]
        d["sent_neg"] = d["sent_neg"]/d["token_count"]
        d["sent_ang"] = d["sent_ang"]/d["token_count"]
        d["sent_ant"] = d["sent_ant"]/d["token_count"]
        d["sent_dis"] = d["sent_dis"]/d["token_count"]
        d["sent_fear"] = d["sent_fear"]/d["token_count"]
        d["sent_joy"] = d["sent_joy"]/d["token_count"]
        d["sent_sad"] = d["sent_sad"]/d["token_count"]
        d["sent_surp"] = d["sent_surp"]/d["token_count"]
        d["sent_trust"] = d["sent_trust"]/d["token_count"]

    return d


#------------------------------------------------------------------------------
path_to_NRC_lexicon = "./lexicons/NRC_combined.xlsx"

# path_to_posts_df = "./dataFiles/kaggle/df_kaggle.pickle"
# path_to_tmp_save = "./NRC_features_tmp.pickle"
# path_to_final_save = "./features/kaggle/NRC_df_kaggle.pickle"

path_to_posts_df = "./dataFiles/reddit/reddit_df_1000.pickle"
path_to_tmp_save = "./NRC_features_tmp.pickle"
path_to_final_save = "./features/reddit/NRC_df_reddit1000.pickle"


nlp = spacy.load("en_core_web_sm")

print("Loading posts df")
with open(path_to_posts_df, "rb") as f:
    df = pickle.load(f)

print("reading NRC dataframe")
NRC_df = pd.read_excel(path_to_NRC_lexicon)

print("creatng NRC dict...")
NRC_dict = NRC_df.set_index("Word").T.to_dict()

# check if we can carry on from where we left off
NRC_features = []
start_idx = 0
if os.path.exists(path_to_tmp_save):
    print("An exisiting NRC features list was found, loading pickle.")
    with open(path_to_tmp_save, "rb") as f:
        NRC_features = pickle.load(f)
        start_idx = len(NRC_features)

print("Computing NRC features...")
its = 0
for i, sentence in enumerate(df["posts"][start_idx:]):

    d = calc_NRC_vals(preprocess_string(sentence))
    d["idx"] = start_idx + i
    NRC_features.append(d)

    its += 1
    if its == 50:
        print(f"{i}. {len(NRC_features)}")
        its = 0

        # save temporary progress
        with open(path_to_tmp_save, "wb") as f:
            pickle.dump(NRC_features, f)


# save final list as df
NRC_df = pd.DataFrame(NRC_features)
NRC_df = NRC_df.fillna(0)

print(f"saving file to {path_to_final_save}")
with open(path_to_final_save, "wb") as f:
    pickle.dump(NRC_df, f)
