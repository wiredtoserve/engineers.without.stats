import numpy as np
import pandas as pd

import nltk
import re
import string

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def get_types(row):
    '''

    :param row: pass the mbti type
    converts the type into a series and returns 4 one hot encodings of each type
    :return: one hot encoding of each type
    '''
    t = row['type']

    I = 0
    N = 0
    T = 0
    J = 0

    if t[0] == 'I':
        I = 1
    elif t[0] == 'E':
        I = 0
    else:
        print('I-E incorrect')

    if t[1] == 'N':
        N = 1
    elif t[1] == 'S':
        N = 0
    else:
        print('N-S incorrect')

    if t[2] == 'T':
        T = 1
    elif t[2] == 'F':
        T = 0
    else:
        print('T-F incorrect')

    if t[3] == 'J':
        J = 1
    elif t[3] == 'P':
        J = 0
    else:
        print('J-P incorrect')
    return pd.Series({'IE': I, 'NS': N, 'TF': T, 'JP': J})


# processing the data
def preprocess_string(x_str,
                      lower_chars=True,
                      rem_kaggle_sep=True,
                      rem_http=True,
                      rem_punctuation=True,
                      rem_numbers=True,
                      stem="lemmatize",
                      rem_stop_words=True,
                      rem_MBTI=True,
                      return_joined=True):
    '''Returns a cleaned string specifically from the MBTI dataset.
    If return_joined is True, the tokens are joined into a single string so
    that it can be passed into SciKit learn's frequency counter. Otherwise
    the tokens are returned as a list.'''

    # lower
    if lower_chars:
        x_str = x_str.lower()

    # remove |||
    if rem_kaggle_sep:
        x_str = re.sub("[]|||[]", " ", x_str)

    # remove http links
    if rem_http:
        x_str = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', x_str)

    # remove puncuation
    if rem_punctuation:
        x_str = "".join([ci for ci in x_str if ci not in string.punctuation])

    # remove numbers
    if rem_numbers:
        x_str = re.sub("[0-9]+", "", x_str)

    # tokenise
    tokens = nltk.word_tokenize(x_str)

    # stem
    if stem is "stem":
        porter = nltk.PorterStemmer()
        tokens = [porter.stem(token) for token in tokens]
    elif stem is "lemmatize":
        # lemmatize
        lemm = WordNetLemmatizer()
        tokens = [lemm.lemmatize(token) for token in tokens]

    # remove stop words
    if rem_stop_words:
        tokens = [ti for ti in tokens if ti not in stopwords.words("english")]

    # remove MBTI types
    if rem_MBTI:
        MBTI_types = ['INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFJ', 'INFP', 'ENFP',
                      'ISFP', 'ISTP', 'ISFJ', 'ISTJ', 'ESTP', 'ESFP', 'ESTJ', 'ESFJ',
                      'MBTI']
        MBTI_types = [ti.lower() for ti in MBTI_types]

        tokens = [wi for wi in tokens if wi not in MBTI_types]

    if return_joined:
        return " ".join([ci for ci in tokens])

    return tokens


def preprocessing(run=False):
    '''
    :param run: user provides if the data processing needs to be run again
    :return: a dataframe with the encoded types and the processed comments
    '''

    if run:
        df = pd.read_csv('dataFiles/toy_data.csv')
        df['processed_post'] = df['posts'].apply(lambda x: preprocess_string(x, True))
        #df = df.merge(df.apply(lambda row: get_types(row), axis=1))
    else:
        df = pd.read_csv('dataFiles/dataFile.csv')

    return df


def avg_word_length(x_str):
    """Returns the average word length (number of chars) of a text string"""

    terms = x_str.split()
    return sum([len(term) for term in terms])/len(terms)
