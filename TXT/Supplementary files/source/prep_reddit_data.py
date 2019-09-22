import pandas as pd
import pickle
import numpy as np


print("Reading in reddit dataset...")
df = pd.read_csv("./dataFiles/reddit/original/mbti9k_comments.csv")

df = df[["comment", "type"]]


# shorten comments to MAX_WORDS
print("Shortening comments...")
MAX_WORDS = 1000
short_comments = []
its = 0
for i, comments in enumerate(df["comment"]):
    word_count = 0
    shortened = []
    for sentence in comments.split("."):
        num_words = len(sentence.split(" "))
        word_count += num_words
        shortened.append(sentence)

        if word_count > MAX_WORDS:
            break

    if its == 500:
        print(f"{i}, {len(shortened)}")
        its = 0

    short_comments.append(".".join(shortened))

# overwrite old comments to reduce space
df["posts"] = short_comments
df = df.drop(columns=["comment"])

# encode MBTI types
print("Encoding types as binary")
df["IE"] = np.array(df["type"].str.contains("i"), dtype=int)
df["NS"] = np.array(df["type"].str.contains("n"), dtype=int)
df["TF"] = np.array(df["type"].str.contains("t"), dtype=int)
df["JP"] = np.array(df["type"].str.contains("j"), dtype=int)

# saving to disk
print("Saving to disk")
with open(f"./dataFiles/reddit/reddit_df_{MAX_WORDS}.pickle", "wb") as f:
    pickle.dump(df, f)
df.to_csv(f"./dataFiles/reddit/reddit_{MAX_WORDS}_coded.csv", index=False)

print("done")
