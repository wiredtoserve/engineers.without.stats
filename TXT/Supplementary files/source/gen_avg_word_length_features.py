
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


def word_lengths(sentence):
    return [len(word) for word in sentence_words(sentence)]


def avg_word_len(sentence_list):
    """Compute the average word length sentence from a list of sentences"""

    count = 0
    for sent in sentence_list:
        wl = word_lengths(sent)
        if len(wl) > 0:
            count += sum(wl)/len(wl)

    return count/len(sentence_list)
#-------------------------------------------------------------------------------


path_to_posts_df = "./datafiles/kaggle/df_kaggle.pickle"
path_to_tmp_save = "./avg_word_len_tmp.pickle"
path_to_final_save = "./features/kaggle/avg_word_len_df.pickle"

posts_df = load_file(path_to_posts_df)



start_idx = 0
word_len_list = []
if os.path.exists(path_to_tmp_save):
    print("Found an existing save file, loading...")
    word_len_list = load_file(path_to_tmp_save)
    start_idx = len(word_len_list)


print("Computing average word length")
its = 0
for i, person in enumerate(posts_df["posts"][start_idx:]):
    posts_list = person.split("|||")

    word_len_list.append((start_idx + i, avg_word_len(posts_list)))

    its += 1
    if its == 50:
        save_file(word_len_list, path_to_tmp_save)
        print(i, len(word_len_list))

        its = 0

df = pd.DataFrame(word_len_list, columns=["idx", "avg word length"])
save_file(df, path_to_final_save)
