from sklearn import metrics
from common import load_file
import pandas as pd
import numpy as np

clf_types = ["MCC", "LR", "NB", "RF", "SVM"]

X_test = load_file("./models/all_features/X_test_kaggle.pickle")
Ys = load_file("./models/all_features/Ys_kaggle.pickle")

scores = {}
scores["accuracy"] = []
scores["recall_1"] = []
scores["precision_1"] = []
scores["f1"] = []
scores["macro"] = []
scores["micro"] = []
scores["weighted"] = []
scores["recall_0"] = []
scores["precision_0"] = []
for clf_type in clf_types:
    print(f"running {clf_type}")


    y_true = Ys["IE_test"]
    if clf_type == "MCC":
        L = len(Ys["IE_test"])
        y_pred = np.ones(L, dtype=int) * round(Ys["IE_train"].mean())
    else:
        # clf = load_file(f"./models/{clf_type}_clf_IE_kaggle.pickle")
        clf = load_file(f"./models/all_features/{clf_type}_clf_IE_kaggle.pickle")
        y_pred = clf.predict(X_test)

    scores["accuracy"].append(metrics.accuracy_score(y_true, y_pred))
    scores["recall_1"].append(metrics.recall_score(y_true, y_pred, pos_label=1))
    scores["precision_1"].append(metrics.precision_score(y_true, y_pred, pos_label=1))
    scores["f1"].append(metrics.f1_score(y_true, y_pred, pos_label=1))
    scores["macro"].append(metrics.f1_score(y_true, y_pred, pos_label=1, average="macro"))
    scores["micro"].append(metrics.f1_score(y_true, y_pred, pos_label=1, average="micro"))
    scores["weighted"].append(metrics.f1_score(y_true, y_pred, pos_label=1, average="weighted"))

    scores["recall_0"].append(metrics.recall_score(y_true, y_pred, pos_label=0))
    scores["precision_0"].append(metrics.precision_score(y_true, y_pred, pos_label=0))

# save to df
df = pd.DataFrame(scores, index=clf_types)
