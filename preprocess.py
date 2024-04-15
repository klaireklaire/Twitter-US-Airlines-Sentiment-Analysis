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
from sklearn.feature_extraction.text import TfidVectorizer

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
    clean_doc = clean_doc.apply(lambda x : " ".join(stemmer.stem(word) for word in x.split()))

    
    return clean_doc

def vectorize(text, is_train = False):
    vectorizer = TfidVectorizer(stop_words='english')
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

def main():

    # df['processed_text'] = preprocess(df['text'])
    # full_text = " ".join(df['processed_text'])
    # doc = nlp(full_text)

    # df.to_pickle("cleaned.csv")
    df_cleaned = pd.read_pickle("cleaned.csv")
    X = vectorize(df_cleaned['processed_text'])
    y = df_cleaned['airline_sentiment']

    # 70% for training, 30% for validation
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=42)

    orig_stdout = sys.stdout
    f = open('out.txt', 'w')
    sys.stdout = f
    
    for sen in df_cleaned['processed_text'] :
        print(sen + "\n")
    
    sys.stdout = orig_stdout
    f.close()
    
    
if __name__ == "__main__":
    main()