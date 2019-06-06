import csv
import sys
sys.path.append("../dataProcessing")
import data_processing as dp

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# read in csv data
with open("./dataFile.csv", 'r', encoding="utf-8") as f:
    itr = csv.reader(f)
    next(itr)

    # create a dictionary to store all the columns in the csv file
    d = {}
    types = []
    posts = []
    IE = []
    NS = []
    TF = []
    JP = []
    processed_posts = []
    for row in itr:
        type, post, dummy_IE, dummy_NS, dummy_TF, dummy_JP, processed_post = row
        types.append(type)
        posts.append(post)
        IE.append(int(dummy_IE))
        NS.append(int(dummy_NS))
        TF.append(int(dummy_TF))
        JP.append(int(dummy_JP))
        processed_posts.append(processed_post)

    d["types"] = types
    d["posts"] = posts
    d["IE"] = IE
    d["NS"] = NS
    d["TF"] = TF
    d["JP"] = JP
    d["processed_posts"] = processed_posts


# count term frequencies
vec = CountVectorizer()
freq_mat = vec.fit_transform(d["processed_posts"])

# calc tfidf matrix
tfidf = TfidfTransformer()
tfidf_mat = tfidf.fit_transform(freq_mat)

# split into train and test data
TRAIN_FRACTION = 0.70
X_train, X_test = train_test_split(tfidf_mat, train_size=TRAIN_FRACTION, random_state=101)

# logistic regression for each bucket
for MBTI_char in ["IE", "NS", "TF", "JP"]:
    Y_train, Y_test = train_test_split(d[MBTI_char], train_size=TRAIN_FRACTION)

    lr = LogisticRegression(multi_class='ovr', solver="lbfgs")
    clf = lr.fit(X_train, Y_train)

    Y_pred = clf.predict(X_test)

    print(f"{MBTI_char}: {accuracy_score(Y_test, Y_pred):.4f}")
    print(confusion_matrix(Y_test, Y_pred))
