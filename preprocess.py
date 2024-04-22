#!/usr/bin/python3

import nltk
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import pandas as pd
import sys
import re
from assets import contractions
from bs4 import BeautifulSoup
from textblob import TextBlob
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, roc_auc_score, roc_curve, f1_score, auc, accuracy_score
import matplotlib.pyplot as plt

nlp = spacy.load('en_core_web_sm')
doc = pd.read_csv('Tweets.csv')
df = pd.DataFrame(doc)
stemmer = SnowballStemmer("english")

def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")                    
    return soup.get_text()

def cont_to_exp(x):
    if type(x) is str:
        for key in contractions:
            value = contractions[key]
            x = x.replace(key, value)
        return x
    else:
        return x

def base_word(x):
    doc = nlp(x)
    lis = []
    for token in doc:
        lemma = token.lemma_
        if lemma == '-PRON-' or lemma == 'be':
            lemma = token.text
        lis.append(lemma)
    return " ".join(lis)


def preprocess(doc):
    #lowercase
    clean_doc = doc.str.lower()
    
    #remove usernames and emails and hashtags
    clean_doc = clean_doc.apply(lambda x : re.sub(r'([A-Za-z0-9+_]+@[A-Za-z0-9+_]+\.[A-Za-z0-9+_]+)','', x))
    clean_doc = clean_doc.apply(lambda x : re.sub(r'@\w+', '', x))
    clean_doc = clean_doc.apply(lambda x : re.sub(r'#\w+', '', x))
    
     # removing URL
    clean_doc = clean_doc.apply(lambda i : re.sub(r'https?://[^\s<>"]+|www\.[^\s<>"]+|http?://[^\s<>"]+','', i))
    # remove HTML tags
    clean_doc = clean_doc.apply(lambda x: strip_html(x))
    
    # Handle punctuation within words
    clean_doc = clean_doc.apply(lambda x : cont_to_exp(x))
    
    # remove no-alphanumeric
    clean_doc = clean_doc.apply(lambda x : re.sub(r'[^\w\s]', '', x))
    
    # remove stopwords
    clean_doc = clean_doc.apply(lambda x : " ".join(t for t in x.split() if t not in STOP_WORDS))
    
    # change text to base words
    # clean_doc = clean_doc.apply(lambda x : base_word(x))

    # stem tokens
    #clean_doc = clean_doc.apply(lambda x : " ".join(stemmer.stem(word) for word in x.split()))

    
    return clean_doc

def vectorize(text, is_train = False):
    vectorizer = TfidfVectorizer(stop_words='english')
    if is_train:
        vectorize_text = vectorizer.fit_transform(text).toarray()
    else:
        vectorize_text = vectorizer.transform(text).toarray()
    return vectorize_text



# print(doc.type())

def tune_train_evaluate_mnb_muticlass(X, y, X_train, X_test, y_train, y_test):
    nb = MultinomialNB()

    param_grid_nb = [
        {'alpha': [1.0e-10, 0.01, 0.1, 0.5, 1.0, 2.0, 10, 20, 50, 100],
         'fit_prior': [True, False]}
    ]

    clf_nb = GridSearchCV(nb, param_grid=param_grid_nb, cv=5, verbose=1, n_jobs=-1, scoring='f1_macro')
    best_clf_nb = clf_nb.fit(X, y)

    print(best_clf_nb.best_score_, best_clf_nb.best_estimator_)

    df_best = pd.DataFrame(best_clf_nb.cv_results_)
    print("Average cross-validation F1 score for all combinations: " + str(df_best.loc[:, 'mean_test_score'].mean()))

    nb_best = best_clf_nb.best_estimator_
    nb_best.fit(X_train, y_train)
    y_predicted = nb_best.predict(X_test)

    ConfusionMatrixDisplay.from_predictions(y_test, y_predicted)
    plt.show()

    print("F1 score:", str(f1_score(y_test, y_predicted, average='macro')))
    print("Accuracy", str(accuracy_score(y_test, y_predicted)))

    return nb_best


# print(nlp)
# def get_text():
#     doc = pd.read_csv('Tweets.csv')
    
#     return doc['text']

def main():

    df['processed_text'] = preprocess(df['text'])
    full_text = " ".join(df['processed_text'])
    doc = nlp(full_text)

    df.to_pickle("cleaned_no_stem.csv")
    df.to_csv("cleaned_readable.csv")
    df_cleaned = pd.read_pickle("cleaned_no_stem.csv")
    print(df_cleaned.head(10)['processed_text'])
    X = vectorize(df_cleaned['processed_text'], is_train=True)
    y = df_cleaned['airline_sentiment']

    # 70% for training, 30% for validation
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=42)

    nb_best = tune_train_evaluate_mnb_muticlass(X, y, xtrain, xtest, ytrain, ytest)
    orig_stdout = sys.stdout
    f = open('out.txt', 'w')
    sys.stdout = f
    
    for sen in df_cleaned['processed_text'] :
        print(sen + "\n")
    
    sys.stdout = orig_stdout
    f.close()
    
    
if __name__ == "__main__":
    main()