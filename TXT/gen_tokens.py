import sklearn as sk

import pickle
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import re

import os

from common import tokeniser


# #-------------------------------------------------------------------------------
# def tokeniser(sentence):
#
#     # Remove ||| from kaggle dataset
#     sentence = re.sub("[]|||[]", " ", sentence)
#
#     # remove reddit subreddit urls
#     sentence = re.sub("/r/[0-9A-Za-z]", "", sentence)
#
#     # remove MBTI types
#     MBTI_types = ['INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFJ', 'INFP', 'ENFP',
#               'ISFP', 'ISTP', 'ISFJ', 'ISTJ', 'ESTP', 'ESFP', 'ESTJ', 'ESFJ',
#               'MBTI']
#     MBTI_types = [ti.lower() for ti in MBTI_types] + [ti.lower() + 's' for ti in MBTI_types]
#
#     tokens = nlp(sentence)
#
#     tokens = [ti for ti in tokens if ti.lower_ not in STOP_WORDS]
#     tokens = [ti for ti in tokens if not ti.is_space]
#     tokens = [ti for ti in tokens if not ti.is_punct]
#     tokens = [ti for ti in tokens if not ti.like_num]
#     tokens = [ti for ti in tokens if not ti.like_url]
#     tokens = [ti for ti in tokens if not ti.like_email]
#     tokens = [ti for ti in tokens if ti.lower_ not in MBTI_types]
#
#     # lemmatize
#     tokens = [ti.lemma_ for ti in tokens if ti.lemma_ not in STOP_WORDS]
#     tokens = [ti for ti in tokens if len(ti) > 1]
#
#     return tokens
#
#
#-------------------------------------------------------------------------------
# path_to_posts_df = "./dataFiles/kaggle/df_kaggle.pickle"

path_to_tmp_save = "./doc_tokens_tmp.pickle"

# path_to_posts_df = "./dataFiles/reddit/reddit_df_500.pickle"
# path_to_final_save = "./features/reddit/reddit_500_tokens.pickle"
path_to_posts_df = "./dataFiles/reddit/reddit_df_1000.pickle"
path_to_final_save = "./features/reddit/reddit_1000_tokens.pickle"


# with open("./dataFiles/df_kaggle.pickle", "rb") as f:
with open(path_to_posts_df, "rb") as f:
    df_posts = pickle.load(f)

print(f"Loaded {path_to_posts_df}")


nlp = spacy.load("en_core_web_sm")


docs_as_tokens = []
start_idx = 0
if os.path.exists(path_to_tmp_save):
    print("An exisiitng tokenised docs list was found, loading")

    with open(path_to_tmp_save, "rb") as f:
        docs_as_tokens = pickle.load(f)

    start_idx = len(docs_as_tokens)

print("Tokenising documents")
its = 0
for i, post in enumerate(df_posts["posts"][start_idx:]):
    docs_as_tokens.append(tokeniser(post))

    its += 1
    if its == 50:
        print(f"iter {i}, {start_idx + i} {len(docs_as_tokens)}")

        # save current progress
        with open(path_to_tmp_save, "wb") as f:
            pickle.dump(docs_as_tokens, f)

        its = 0

# save to final path
print(f"Saving tokens to {path_to_final_save}")
with open(path_to_final_save, "wb") as f:
    pickle.dump(docs_as_tokens, f)
