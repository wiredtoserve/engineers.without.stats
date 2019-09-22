import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from common import load_file, save_file

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.model_selection import train_test_split

# feature scaling
from sklearn.preprocessing import StandardScaler

# metrics and scoring
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score

# gridsearch
from sklearn.model_selection import GridSearchCV

#classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB


#-------------------------------------------------------------------------------
# used to bypass sklearn's internal tokeniser
# dummy_fn = lambda x: x
def dummy_fn(x): return x


def build_Y_dims(df_list):
    """vstack all dimensions into four separate arrays
    IE, NS, TF and JP
    """

    Y_IE = df_list[0]["IE"].values
    Y_NS = df_list[0]["NS"].values
    Y_TF = df_list[0]["TF"].values
    Y_JP = df_list[0]["JP"].values
    for i in range(1, len(df_list)):
        df = df_list[i]
        Y_IE = np.append(Y_IE, df['IE'].values)
        Y_NS = np.append(Y_NS, df['NS'].values)
        Y_TF = np.append(Y_TF, df['TF'].values)
        Y_JP = np.append(Y_JP, df['JP'].values)

    return (Y_IE, Y_NS, Y_TF, Y_JP)


#-------------------------------------------------------------------------------
# posts files
path_to_kaggle_posts_df = "./dataFiles/kaggle/df_kaggle.pickle"
# path_to_reddit_posts_df = "./dataFiles/reddit/reddit_df_500.pickle"
path_to_reddit_posts_df = "./dataFiles/reddit/reddit_df_1000.pickle"

# token files
path_to_kaggle_tokens = "./features/kaggle/tokens_kaggle.pickle"
# path_to_reddit_tokens = "./features/reddit/reddit_500_tokens.pickle"
path_to_reddit_tokens = "./features/reddit/reddit_1000_tokens.pickle"

# NRC files
path_to_kaggle_NRC = "./features/kaggle/NRC_features_kaggle.pickle"
path_to_reddit_NRC = "./features/reddit/NRC_df_reddit1000.pickle"

# TTR files
path_to_kaggle_ttr = "./features/kaggle/ttrs_df_kaggle.pickle"
path_to_reddit_ttr = "./features/reddit/ttrs_df_reddit1000.pickle"


#-------------------------------------------------------------------------------

# load kaggle and reddit datasets for ys
df_kaggle = load_file(path_to_kaggle_posts_df)
df_reddit = load_file(path_to_reddit_posts_df)

# Y_IE, Y_NS, Y_TF, Y_JP = build_Y_dims([df_kaggle, df_reddit])
Y_IE, Y_NS, Y_TF, Y_JP = build_Y_dims([df_kaggle])
# Y_IE, Y_NS, Y_TF, Y_JP = build_Y_dims([df_reddit])

# glue tokens from kaggle and reddit together
kaggle_tokens = load_file(path_to_kaggle_tokens)
reddit_tokens = load_file(path_to_reddit_tokens)

# all_tokens = kaggle_tokens + reddit_tokens
all_tokens = kaggle_tokens
# all_tokens = reddit_tokens

# build count vectoriser and create counts matrix
# unigrams seem to work the best, max_features>2000 tends to give worse accuracy
cv = CountVectorizer(tokenizer=dummy_fn,
                     preprocessor=dummy_fn,
                     ngram_range=(1, 1),
                     max_features=2000)
                     # max_df = 0.7,
                     # min_df=0.1)
cv.fit(all_tokens)
tf_matrix = cv.transform(all_tokens)

# create td-idf matrix
idf_transformer = TfidfTransformer(smooth_idf=True)
idf_transformer.fit(tf_matrix)
tfidf_matrix = idf_transformer.transform(tf_matrix)

# load NRC features
with open(path_to_kaggle_NRC, "rb") as f:
    NRC_kaggle_df = pickle.load(f)

with open(path_to_reddit_NRC, "rb") as f:
    NRC_reddit_df = pickle.load(f)

NRC_colnames = ["val",
                "arou",
                "dom",
                "sent_ang",
                "sent_dis",
                "sent_fear",
                "sent_joy",
                "sent_sad",
                "sent_surp",
                "sent_trust",
                "sent_neg",
                "sent_pos"]

NRC_features = NRC_kaggle_df[NRC_colnames].values
# NRC_features = NRC_reddit_df[NRC_colnames].values

# load type-token ratio feature
with open(path_to_kaggle_ttr, "rb") as f:
    kaggle_ttr_df = pickle.load(f)

with open(path_to_reddit_ttr, "rb") as f:
    reddit_ttr_df = pickle.load(f)

TTRs = np.array([kaggle_ttr_df["TTR"].values]).T  # convert to 2-d vector so that it can be hstacked
# TTRs = np.array([reddit_ttr_df["TTR"].values]).T  # convert to 2-d vector so that it can be hstacked

path_to_kaggle_emoticons = "./features/kaggle/emoticon_counts_df.pickle"
path_to_kaggle_formality = "./features/kaggle/formality_df.pickle"
emoticons = load_file(path_to_kaggle_emoticons)
emoticons = np.array([emoticons["emoticons per post"].values]).T
formalities = load_file(path_to_kaggle_formality)
formalities = np.array([formalities["avg formality"].values]).T


path_to_kaggle_wps = "./features/kaggle/avg_wps_df.pickle"
path_to_kaggle_word_len = "./features/kaggle/avg_word_len_df.pickle"
words_per_sent = load_file(path_to_kaggle_wps)
words_per_sent = np.array([words_per_sent["avg wps"].values]).T

word_len = load_file(path_to_kaggle_word_len)
word_len = np.array([word_len["avg word length"].values]).T

path_to_kaggle_http = "./features/kaggle/http_df.pickle"
https = load_file(path_to_kaggle_http)
https = np.array([https["http_frac"].values]).T


#-------------------------------------------------------------------------------

# glue all features together
# X = np.hstack([tfidf_matrix.todense(), NRC_features, TTRs])
# X = np.hstack([tfidf_matrix.todense(), NRC_features])
# X = np.hstack([tfidf_matrix.todense(), emoticons])
X = np.hstack([tfidf_matrix.todense(), NRC_features, TTRs, emoticons, formalities, words_per_sent, word_len, https])
# X = tfidf_matrix.todense()
# X = NRC_features
# X = np.hstack([NRC_features, TTRs])

# feature scaling
# Note: scaling tends to degrade accuracy and f1-score, but improves recall - makes predictions more balanced with more predictions of 0, so it might be less prone to overfitting and bias
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# train-test split
test_size = 0.3
X_train,X_test, Y_IE_train,Y_IE_test, Y_NS_train,Y_NS_test, Y_TF_train,Y_TF_test, Y_JP_train,Y_JP_test = train_test_split(X, Y_IE, Y_NS, Y_TF, Y_JP, test_size=test_size, random_state=1)

Ys = {}
Ys["IE_train"] = Y_IE_train
Ys["NS_train"] = Y_NS_train
Ys["TF_train"] = Y_TF_train
Ys["JP_train"] = Y_JP_train

Ys["IE_test"] = Y_IE_test
Ys["NS_test"] = Y_NS_test
Ys["TF_test"] = Y_TF_test
Ys["JP_test"] = Y_JP_test


#-------------------------------------------------------------------------------

lr_classifier = LogisticRegression(multi_class="ovr",
                                   solver="lbfgs",
                                   class_weight={1:1, 0:1})  # default
                                   # class_weight={1:1, 0:3.5})  # opt IE weights
                                   # class_weight={1:1, 0:6})   # opt NS
                                   # class_weight={1:1, 0:1})   # opt TF
                                   # class_weight={1:1.5, 0:1}) # opt JP

rf_classifier = RandomForestClassifier(n_estimators=100,
                                       criterion="gini",
                                       # max_features=50,
                                       max_features='auto',
                                       max_depth=None,
                                       random_state=1)

svm_classifier = SVC(random_state=1,
                     probability=True,
                     gamma='scale')

# Bernoulli NB
nb_classifier = BernoulliNB()


#-------------------------------------------------------------------------------
# select classifier and train

clf = lr_classifier
# clf = rf_classifier
# clf = svm_classifier   # SVM tends to do better WITHOUT tokens
# clf = nb_classifier    # terrible without tokens, but not bad with 2k n-grams

mbti_type = "IE"


print(f"Fitting classifier to {mbti_type} type...")
clf.fit(X_train, Ys[mbti_type + "_train"])

# print classification report
y_true = Ys[mbti_type + "_test"]
y_pred = clf.predict(X_test)
y_probs = clf.predict_proba(X_test)[:, 1].flatten()
y_mc_probs = np.ones(len(y_true))*round(np.mean(y_true))  # majority class predictions

print(classification_report(y_true, y_pred))
print("Confusion matrix:")
print(confusion_matrix(y_true, y_pred).T)
auc = roc_auc_score(y_true, y_probs)
print(f"AUC = {auc:.4f}")


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# select classifier and train

# clf = lr_classifier
# clf = rf_classifier
clf = svm_classifier   # SVM tends to do better WITHOUT tokens
# clf = nb_classifier    # terrible without tokens, but not bad with 2k n-grams

for mbti_type in ["IE", "NS", "TF", "JP"]:


    print(f"Fitting classifier to {mbti_type} type...")
    clf.fit(X_train, Ys[mbti_type + "_train"])

    # print classification report
    y_true = Ys[mbti_type + "_test"]
    y_pred = clf.predict(X_test)
    y_probs = clf.predict_proba(X_test)[:, 1].flatten()
    y_mc_probs = np.ones(len(y_true))*round(np.mean(y_true))  # majority class predictions

    print(classification_report(y_true, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred).T)
    auc = roc_auc_score(y_true, y_probs)
    print(f"AUC = {auc:.4f}")

    save_file(clf, f"./models/all_features/SVM_clf_{mbti_type}_kaggle.pickle")

save_file(X_test, "./models/all_features/X_test.pickle")
save_file(Ys, "./models/all_features/Ys.pickle")


#-------------------------------------------------------------------------------
# plot ROC curve
fpr, tpr, _ = roc_curve(y_true, y_probs)
fpr_mc, tpr_mc, _ = roc_curve(y_true, y_mc_probs)

fig = plt.figure()
ax = fig.gca()
ax.plot(fpr, tpr, 'b-')
ax.plot(fpr_mc, tpr_mc, 'go-')
ax.plot([0,1], [0,1], 'k--')  # diagonal line

ax.set_xlabel("FPR")
ax.set_ylabel("TPR")
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])


#-------------------------------------------------------------------------------
# from common import tokeniser
# sample_text = '''I love meeting new people and experiencing new things i love to everyone and everything the world is a beautiful place and life is amazing'''
#
# c = cv.transform([tokeniser(sample_text)])
# x = idf_transformer.transform(c).todense()
#
# print(clf.predict_proba(x))

#-------------------------------------------------------------------------------
# define  classifiers
# print("Grid searching...")
#
# parameters = {'class_weight': [{0:1, 1:1},
#                                {0:1, 1:2},
#                                {0:1, 1:3},
#                                {0:1, 1:4},
#                                {0:1, 1:5},
#                                {0:1, 1:6},
#                                {0:1, 1:7},
#                                {0:1, 1:8},
#                                {0:1, 1:9},
#                                {0:1, 1:10}]
# }
# # parameters = {"class_weight": [{1:1, 0:2 + x/1} for x in range(11)]}
#
# lr_classifier = LogisticRegression(multi_class="ovr",
#                                    solver="lbfgs",
#                                    max_iter=200)
#
# gridsearch_clf = GridSearchCV(lr_classifier,
#                    parameters,
#                    cv=5,
#                    n_jobs = -1,
#                    scoring="recall")
#
# gridsearch_clf.fit(X_train, Ys["IE_train"])
# # clf.best_estimator
