import spacy
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import numpy as np


nlp = spacy.load("en_core_web_lg")


docs = [nlp(":D").vector,
        nlp("lol").vector,
        nlp("fun").vector,
        nlp("guy").vector,
        nlp("awesome").vector,
        nlp("friend").vector,
        nlp("crazy").vector,

        nlp("feel").vector,
        nlp("love").vector,
        nlp("beautiful").vector,
        nlp("thank").vector,
        nlp("argument").vector,
        nlp("happy").vector,]


Xs = np.vstack(docs)

model = MDS(n_components=2, random_state=2)
outS = model.fit_transform(Xs)
plt.plot(outS[:7, 0], outS[:7, 1], "bo")
plt.plot(outS[7:, 0], outS[7:, 1], "ro")
# plt.axis('equal');

plt.show()
