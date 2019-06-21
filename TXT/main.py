# author: Syndicate 7
# Project: Text and Web Based Analytics

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import nltk
import re
import string

from data_processing import preprocessing
from classifier import pre_classification, multinomial, majority_classifier, random_forrest, support_vector, \
    logistic_regression

import warnings

warnings.filterwarnings('ignore')


def main():
    '''
    Step 1: Run the pre-processor
    Step 2: Run different classifiers based off the input
    TODO: Add parameters to pass via the main funciton, eg: Kernel function in SVM

    Step 3: Output the results of these classifiers

    TODO: Find a way to represent the output on some sort of scale

    --------------------
    Step 4: Scrape the web-data for information about the companies
    Step 5: Map the companies with our best classifier
    Step 6: Activate Voice Recorder or in turn collect free text
    Step 7: Display the results in terms of the users personality scale and also best cultural fit

    TODO: Alternatively we could just focus on the Big 4 firms and see which candidates are the best match

    :output: the summary of statistics of the classifiers
    '''

    # Step 1: pass parameter True if you want to compute the processor again
    print('Starting pre-processing... \n')
    df = preprocessing(True)

    # Step 2
    X_train, X_test, y_ie_train, y_ie_test, y_ns_train, y_ns_test, y_tf_train, y_tf_test, y_jp_train, y_jp_test = pre_classification(
        df)

    train_label = [y_ie_train, y_ns_train, y_tf_train, y_jp_train]
    test_label = [y_ie_test, y_ns_test, y_tf_test, y_jp_test]

    # dictionary to output the scores
    model_scores = {}

    f, ax = plt.subplots(2, 2)

    print('Trying majority \n')
    model_scores['majority'] = majority_classifier(train_label, test_label, X_test)

    print('Trying multinomial \n')
    model_scores['multinomial'] = multinomial(X_train, X_test, train_label, test_label, ax[1,1])

    print('Trying random forrest \n')
    model_scores['random_forrest'] = random_forrest(X_train, X_test, train_label, test_label, ax[0,1])

    print('Trying support vector \n')
    model_scores['support_vector'] = support_vector(X_train, X_test, train_label, test_label, ax[1,0])

    print('Trying logistic regression \n')
    model_scores['logistic_regression'] = logistic_regression(X_train, X_test, train_label, test_label, ax[0,0])

    plt.show(f)

    summary = pd.DataFrame.from_dict(model_scores, orient='index', columns=['I/E', 'N/S', 'T/F', 'J/P'])

    print('\n \n \n')
    print(summary)


if __name__ == "__main__":
    main()
