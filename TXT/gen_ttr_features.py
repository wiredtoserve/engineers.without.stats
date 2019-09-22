import pickle
from collections import defaultdict
import pandas as pd
import re
import spacy
import os

# remove MBTI types
MBTI_types = ['INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFJ', 'INFP', 'ENFP',
          'ISFP', 'ISTP', 'ISFJ', 'ISTJ', 'ESTP', 'ESFP', 'ESTJ', 'ESFJ',
          'MBTI']
MBTI_types = [ti.lower() for ti in MBTI_types]


def preprocess_string(sentence):
    '''Remove ||| from kaggle dataset'''

    sentence = re.sub("[]|||[]", " ", sentence)

    # remove http
    sentence = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', sentence)

    tmp = [token.text for token in nlp(sentence) if token.lower_ not in MBTI_types]

    return " ".join(tmp)

#-------------------------------------------------------------------------------
# path_to_posts_df = "./dataFiles/kaggle/df_kaggle.pickle"
# path_to_tmp_save = "./ttr_tmp.pickle"
# path_to_final_save = "./features/kaggle/ttrs_df_kaggle.pickle"

path_to_posts_df = "./dataFiles/reddit/reddit_df_1000.pickle"
path_to_tmp_save = "./ttr_tmp.pickle"
path_to_final_save = "./features/reddit/ttrs_df_reddit1000.pickle"


with open(path_to_posts_df, "rb") as f:
    df = pickle.load(f)


nlp = spacy.load("en_core_web_sm")

# check if temp file exists
ttrs = []
start_idx = 0
if os.path.exists(path_to_tmp_save):
    print("found temp save, loading")
    with open(path_to_tmp_save, "rb") as f:
        ttrs = pickle.load(f)
        start_idx = len(ttrs)

print(f"computing ttrs for {path_to_posts_df}...")
its = 0
for i, post in enumerate(df["posts"][start_idx:]):
    processed_post = preprocess_string(post)

    dd = defaultdict(int)
    num_tokens = 0
    for token in nlp(processed_post):
        if not token.is_punct:
            dd[token.lower_] += 1
            num_tokens += 1

    num_unique_tokens = len(dd.keys())
    if num_tokens==0:
        ttr = 0
    else:
        ttr = num_unique_tokens/num_tokens
    ttrs.append(ttr)

    its += 1
    if its == 25:
        print(f"{i}, {len(ttrs)}")

        with open(path_to_tmp_save, "wb") as f:
            pickle.dump(ttrs, f)

        its = 0

# save to df
print(f"saving results to {path_to_final_save}")
ttr_df = pd.DataFrame(ttrs, columns=["TTR"])
with open(path_to_final_save, "wb") as f:
    pickle.dump(ttr_df, f)

print("done")
