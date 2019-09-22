import pandas as pd
import nltk
import spacy
import re
import string
import pickle
import numpy as np

import data_processing as dp
import sklearn as sk

# df = pd.read_csv("./dataFiles/toy_data.csv")
df = pd.read_csv("./dataFiles/kaggle_coded.csv")

# add MBTI dummy vars if needed
# df["IE"] = np.where(df["type"].str.contains("I"), 1, 0)
# df["NS"] = np.where(df["type"].str.contains("N"), 1, 0)
# df["TF"] = np.where(df["type"].str.contains("T"), 1, 0)
# df["JP"] = np.where(df["type"].str.contains("J"), 1, 0)

p = dp.preprocess_string(df["posts"][51], rem_punctuation=False)

vectorizer = sk.feature_extraction.text.CountVectorizer(preprocessor=dp.preprocess_string)
an = vectorizer.build_analyzer()

counts = vectorizer.fit_transform(["hello", "cool cool day", "cool cat", "hello cool winter"])
transformer = sk.feature_extraction.text.TfidfTransformer()

X = transformer.fit_transform(counts)


svc = sk.svm.SVC()
lr = sk.linear_model.LogisticRegression()

Y = df["IE"]

X_train, X_test, Y_train, Y_test = sk.model_selection.train_test_split(X, Y, train_size=0.7)

svc.fit(X_train, Y_train)
