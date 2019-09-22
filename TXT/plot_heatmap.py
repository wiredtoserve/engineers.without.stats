import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import seaborn as sns

from common import load_file


clf_IE = load_file("./models/LR_clf_IE_kaggle.pickle")
clf_NS = load_file("./models/LR_clf_NS_kaggle.pickle")
clf_TF = load_file("./models/LR_clf_TF_kaggle.pickle")
clf_JP = load_file("./models/LR_clf_JP_kaggle.pickle")
X_test = load_file("./models/X_test_kaggle.pickle")
Ys = load_file("./models/Ys_kaggle.pickle")

d = {}
all_mbti_types = ['INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFJ', 'INFP', 'ENFP', 'ISFP', 'ISTP', 'ISFJ', 'ISTJ', 'ESTP', 'ESFP', 'ESTJ', 'ESFJ']
for idx, mbti_type in enumerate(all_mbti_types):
    binary = []
    for c in mbti_type:
        if c=="I" or c=="N" or c=="T" or c=="J":
            binary.append(1)
        else:
            binary.append(0)
    d[mbti_type] = (idx, np.array(binary))



Y_pred = np.array([clf_IE.predict(X_test),
                   clf_NS.predict(X_test),
                   clf_TF.predict(X_test),
                   clf_JP.predict(X_test)]).T

Y_true = np.array([Ys["IE_test"],
                   Ys["NS_test"],
                   Ys["TF_test"],
                   Ys["JP_test"]]).T

# check which rows got all right
diff = Y_pred - Y_true
correct = np.array(~diff.any(axis=1), dtype=int)

ys_pred = []
for y in Y_pred:
    for mbti_type in d.keys():
        if np.all(y == d[mbti_type][1]):
            ys_pred.append(mbti_type)

ys_true = []
for y in Y_true:
    for mbti_type in d.keys():
        if np.all(y == d[mbti_type][1]):
            ys_true.append(mbti_type)

conf_mat = confusion_matrix(ys_true, ys_pred, labels=all_mbti_types)

# from sklearn.cluster import AgglomerativeClustering
# agg = AgglomerativeClustering(affinity="manhattan", linkage="average", n_clusters=2)
# agg.fit(conf_mat)
#
# x = sorted([(val, i) for i,val in enumerate(agg.labels_)])
# k = [xi[1] for xi in x]
# row_labels = [all_mbti_types[ki] for ki in k]
row_labels = all_mbti_types
k = range(16)

fig = plt.figure(figsize=(8,8))
ax = fig.gca()

conf_df = pd.DataFrame(conf_mat[k,:], columns=all_mbti_types, index=row_labels)
sns.heatmap(conf_df, cmap=sns.cubehelix_palette(8), vmax=50, square=True, ax=ax, cbar=False)

# cax = plt.gcf().axes[-1]
# cax.set_yticklabels(["0", "10", "20", "30", "40", "50 or more"])
ax.set_yticklabels(row_labels, fontsize=15, rotation=0)
ax.set_xticklabels(row_labels, fontsize=15, rotation=90)

fig.tight_layout()
