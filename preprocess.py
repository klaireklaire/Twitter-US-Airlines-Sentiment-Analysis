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
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

nlp = spacy.load('en_core_web_sm')

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

vectorizer = TfidfVectorizer(stop_words='english')
def vectorize(text, is_train = False):
    
    if is_train:
        vectorize_text = vectorizer.fit_transform(text).toarray()
    else:
        vectorize_text = vectorizer.transform(text).toarray()
    return vectorize_text



# print(doc.type())


# print(nlp)
# def get_text():
#     doc = pd.read_csv('Tweets.csv')
    
#     return doc['text']

