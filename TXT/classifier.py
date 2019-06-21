from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.pipeline import Pipeline

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


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


def multinomial(X_train, X_test, train_label, test_label):
    for i in range(4):
        print(f'Value of i = {i} ')
        nb_pipeline = Pipeline([
            ('vector', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('Classifier', MultinomialNB())
        ])

        print('Fitting the model...')

        nb_pipeline.fit(X_train, train_label[i])

        predictions = nb_pipeline.predict(X_test)

        print(nb_pipeline.score(X_test, test_label[i]))
