from common import load_file, save_file
import pandas as pd
import spacy
import re
import os


nlp = spacy.load("en_core_web_sm")


def sentence_words(sentence):
    '''returns a list of words that are not urls from a sentence'''

    tokens = [token for token in nlp(sentence) if not token.like_url and not token.is_punct and not token.is_space]
    return tokens


def avg_wps(sentence_list):
    """Compute the average words per sentence from a list of sentences"""

    count = 0
    for sent in sentence_list:
        count += len(sentence_words(sent))

    return count/len(sentence_list)
#-------------------------------------------------------------------------------


path_to_posts_df = "./datafiles/kaggle/df_kaggle.pickle"
path_to_tmp_save = "./avg_wps_tmp.pickle"
path_to_final_save = "./features/kaggle/avg_wps_df.pickle"

posts_df = load_file(path_to_posts_df)

start_idx = 0
wps_list = []
if os.path.exists(path_to_tmp_save):
    wps_list = load_file(path_to_tmp_save)
    start_idx = len(wps_list)

its = 0
for i, person in enumerate(posts_df["posts"][start_idx:]):
    posts_list = person.split("|||")

    wps_list.append((start_idx + i, avg_wps(posts_list)))

    its += 1
    if its == 50:
        save_file(wps_list, path_to_tmp_save)
        print(i, len(wps_list))

        its = 0

df = pd.DataFrame(wps_list, columns=["idx", "avg wps"])
save_file(df, path_to_final_save)
