from common import load_file, save_file
from nltk.tokenize import TweetTokenizer
import re
import pandas as pd
from collections import defaultdict
import os

tkn = TweetTokenizer()
# emoticon_string = r"""
#     (?:
#       [<>]?
#       [:;=8]                     # eyes
#       [\-o\*\']?                 # optional nose
#       [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
#       |
#       [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
#       [\-o\*\']?                 # optional nose
#       [:;=8]                     # eyes
#       [<>]?
#     )"""

emoticon_string = r"""(?:[<>]?[:;=8][\-o\*\']?[\)\]\(\[dDpP/\:\}\{@\|\\]|[\)\]\(\[dDpP/\:\}\{@\|\\][\-o\*\']?[:;=8][<>]?)"""


def identify_emoticons(sentence):
    d = defaultdict(int)
    count = 0
    for token in tkn.tokenize(sentence):
        if re.match(emoticon_string, token):
            d[token] += 1
    return d


def count_emoticons_per_post(posts):

    count = 0
    for post in posts:
        count += len(identify_emoticons(post))

    return count/len(posts)
#-------------------------------------------------------------------------------
path_to_posts_df = "./dataFiles/kaggle/df_kaggle.pickle"
path_to_tmp_save = "./emoticon_tmp.pickle"
path_to_final_save = "./features/kaggle/emoticon_counts_df.pickle"

posts_df = load_file(path_to_posts_df)

start_idx = 0
emoticon_list = []
if os.path.exists(path_to_tmp_save):
    print("found existing tmp save, loading...")
    emoticon_list = load_file(path_to_tmp_save)
    start_idx = len(emoticon_list)

print(f"Calculating emoticons for {path_to_posts_df}...")
its = 0
for i, person in enumerate(posts_df["posts"][start_idx:]):
    idx = start_idx + i

    emoticon_list.append((idx, count_emoticons_per_post(person.split("|||"))))

    its += 1
    if its == 50:
        save_file(emoticon_list, path_to_tmp_save)

        print(f"{idx} {len(emoticon_list)}")
        its = 0


print(f"Saving results to {path_to_final_save}")
df = pd.DataFrame(emoticon_list, columns=["idx", "emoticons per post"])

save_file(df, path_to_final_save)
