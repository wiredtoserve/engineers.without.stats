import re
import string
import nltk
from nltk.corpus import stopwords


def preprocess_string(x_str, return_joined=True):
    '''Returns a cleaned string specifically for the MBTI dataset posts.

    If return_joined is True, the tokens are joined into a single string so
    that it can be passed into SciKit learn's frequency counter. Otherwise
    the tokens are returned as a list.'''

    # lower
    x_str = x_str.lower()

    # remove |||
    x_str = re.sub("[]|||[]", " ", x_str)

    # remove http links
    x_str = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', x_str)

    # remove punctuation
    x_str = "".join([ci for ci in x_str if ci not in string.punctuation])

    # tokenise
    tokens = nltk.word_tokenize(x_str)

    # stem
    porter = nltk.PorterStemmer()
    stemmed_tokens = [porter.stem(token) for token in tokens]

    # remove stop words
    stopped_tokens = [ti for ti in stemmed_tokens if ti not in stopwords.words("english")]

    # remove MBTI types
    MBTI_types = ['INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFJ', 'INFP', 'ENFP',
                  'ISFP', 'ISTP', 'ISFJ', 'ISTJ', 'ESTP', 'ESFP', 'ESTJ', 'ESFJ']
    MBTI_types = [ti.lower() for ti in MBTI_types]

    final_tokens = [wi for wi in stopped_tokens if wi not in MBTI_types]

    if return_joined:
        return " ".join([ci for ci in final_tokens])

    return final_tokens
