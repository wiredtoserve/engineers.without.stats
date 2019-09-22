import pandas as pd
from common import load_file, save_file
import spacy
from collections import defaultdict
import os


def sentence_formality(spacy_sentence):

    d = defaultdict(int)

    for word in spacy_sentence:
        d[word.pos_] += 1

    if len(spacy_sentence) == 0:
        return 0

    return ((d["NOUN"] + d["ADJ"] + d["ADP"] - d["PRON"] + d["DET"] \
            - d["VERB"] - d["ADV"] - d["INTJ"])/len(spacy_sentence) + 100)/2


def avg_formality_of_posts(posts_list):

    score = 0
    for post in posts_list:
        score += sentence_formality(nlp(post))
    score /= len(posts_list)

    return score
#-------------------------------------------------------------------------------

nlp = spacy.load("en_core_web_md")


path_to_posts_df = "./dataFiles/kaggle/df_kaggle.pickle"
path_to_tmp_save = "./formality_tmp.pickle"
path_to_final_save = "./features/kaggle/formality_df.pickle"

posts_df = load_file(path_to_posts_df)

start_idx = 0
formality_list = []
if os.path.exists(path_to_tmp_save):
    print("found existing tmp save, loading...")
    formality_list = load_file(path_to_tmp_save)
    start_idx = len(formality_list)

print(f"Calculating formality for {path_to_posts_df}...")
its = 0
for i, person in enumerate(posts_df["posts"][start_idx:]):
    idx = start_idx + i

    formality_list.append((idx, avg_formality_of_posts(person.split("|||"))))

    its += 1
    if its == 50:
        save_file(formality_list, path_to_tmp_save)

        print(f"{idx} {len(formality_list)}")
        its = 0

print(f"Saving results to {path_to_final_save}")
df = pd.DataFrame(formality_list, columns=["idx", "avg formality"])

save_file(df, path_to_final_save)
