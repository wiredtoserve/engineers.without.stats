import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.pipeline import Pipeline

from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score, auc, roc_curve

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


def pre_classification(df):
    X = df['processed_post']
    y = df['type']

    y_IE = df['IE']
    y_NS = df['NS']
    y_TF = df['TF']
    y_JP = df['JP']

    X_train, X_test, y_ie_train, y_ie_test = train_test_split(X, y_IE, test_size=0.3, random_state=101)
    X_train, X_test, y_ns_train, y_ns_test = train_test_split(X, y_NS, test_size=0.3, random_state=101)
    X_train, X_test, y_tf_train, y_tf_test = train_test_split(X, y_TF, test_size=0.3, random_state=101)
    X_train, X_test, y_jp_train, y_jp_test = train_test_split(X, y_JP, test_size=0.3, random_state=101)

    return X_train, X_test, y_ie_train, y_ie_test, y_ns_train, y_ns_test, y_tf_train, y_tf_test, y_jp_train, y_jp_test


def majority_classifier(train_label, test_label, X_test):
    model_accuracy = []

    for i in range(4):
        val = round(sum(train_label[i]) / len(train_label[i]))
        predictions = [val for i in X_test]
        model_accuracy.append(
            precision_recall_fscore_support(test_label[i], np.array(predictions), average='weighted')[2])
    return model_accuracy


def multinomial(X_train, X_test, train_label, test_label, ax):
    model_accuracy = []

    for i in range(4):
        nb_pipeline = Pipeline([
            ('vector', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('Classifier', MultinomialNB())
        ])

        print(f'Fitting the model NB ... run {i}')

        nb_pipeline.fit(X_train, train_label[i])

        predictions = nb_pipeline.predict(X_test)

        model_accuracy.append(precision_recall_fscore_support(test_label[i], predictions, average='weighted')[2])

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for j in range(2):
            fpr[j], tpr[j], _ = roc_curve(test_label[i], predictions)
            roc_auc[j] = auc(fpr[j], tpr[j])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(test_label[i].ravel(), predictions.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        color_type = ['darkorange', 'red', 'blue', 'green']

        lw = 2
        ax.plot(fpr[1], tpr[1], color=color_type[i],
                lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Multinomial NB ROC ')

    ax.plot([0, 1], [0, 1], color='black', linestyle='--')
    ax.legend(['I/E', 'N/S', 'T/F', 'J/P', 'random'], loc="lower right")

    return model_accuracy


def random_forrest(X_train, X_test, train_label, test_label, ax):
    model_accuracy = []

    for i in range(4):
        rf_pipeline = Pipeline([
            ('vector', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('Classifier', RandomForestClassifier(n_estimators=10))
        ])

        print(f'Fitting the RF model... run {i}')

        rf_pipeline.fit(X_train, train_label[i])

        predictions = rf_pipeline.predict(X_test)

        model_accuracy.append(precision_recall_fscore_support(test_label[i], predictions, average='weighted')[2])

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for j in range(2):
            fpr[j], tpr[j], _ = roc_curve(test_label[i], predictions)
            roc_auc[j] = auc(fpr[j], tpr[j])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(test_label[i].ravel(), predictions.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        color_type = ['darkorange', 'red', 'blue', 'green']

        lw = 2
        ax.plot(fpr[1], tpr[1], color=color_type[i],
                lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Random Forrest ROC ')

    ax.plot([0, 1], [0, 1], color='black', linestyle='--')
    ax.legend(['I/E', 'N/S', 'T/F', 'J/P', 'random'], loc="lower right")

    return model_accuracy


def support_vector(X_train, X_test, train_label, test_label, ax):
    model_accuracy = []

    for i in range(4):
        svc_pipeline = Pipeline([
            ('vector', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('Classifier', SVC(gamma='scale'))
        ])

        print(f'Fitting the SVC model... run {i}')

        svc_pipeline.fit(X_train, train_label[i])

        predictions = svc_pipeline.predict(X_test)

        model_accuracy.append(precision_recall_fscore_support(test_label[i], predictions, average='weighted')[2])

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for j in range(2):
            fpr[j], tpr[j], _ = roc_curve(test_label[i], predictions)
            roc_auc[j] = auc(fpr[j], tpr[j])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(test_label[i].ravel(), predictions.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        color_type = ['darkorange', 'red', 'blue', 'green']

        lw = 2
        ax.plot(fpr[1], tpr[1], color=color_type[i],
                lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Support Vector ROC ')

    ax.plot([0, 1], [0, 1], color='black', linestyle='--')
    ax.legend(['I/E', 'N/S', 'T/F', 'J/P', 'random'], loc="lower right")

    return model_accuracy


def logistic_regression(X_train, X_test, train_label, test_label, ax):
    model_accuracy = []

    for i in range(4):
        lr_pipeline = Pipeline([
            ('vector', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('Classifier', LogisticRegression())
        ])

        print(f'Fitting the Logistic model... run {i}')

        lr_pipeline.fit(X_train, train_label[i])

        predictions = lr_pipeline.predict(X_test)

        model_accuracy.append(precision_recall_fscore_support(test_label[i], predictions, average='weighted')[2])

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for j in range(2):
            fpr[j], tpr[j], _ = roc_curve(test_label[i], predictions)
            roc_auc[j] = auc(fpr[j], tpr[j])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(test_label[i].ravel(), predictions.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        color_type = ['darkorange', 'red', 'blue', 'green']

        lw = 2
        ax.plot(fpr[1], tpr[1], color=color_type[i],
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Logistic Regression ROC ')

    ax.plot([0, 1], [0, 1], color='black', linestyle='--')
    ax.legend(['I/E', 'N/S', 'T/F', 'J/P', 'random'],loc="lower right")

    return model_accuracy
