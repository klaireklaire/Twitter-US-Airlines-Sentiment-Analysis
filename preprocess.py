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
nlp = spacy.load('en_core_web_sm')


doc = pd.read_csv('Tweets.csv')
df = pd.DataFrame(doc)

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
    lower_doc = doc.str.lower()
    
    #remove usernames and emails and hashtags
    lower_doc = lower_doc.apply(lambda x : re.sub(r'([A-Za-z0-9+_]+@[A-Za-z0-9+_]+\.[A-Za-z0-9+_]+)','', x))
    lower_doc = lower_doc.apply(lambda x : re.sub(r'@\w+', '', x))
    lower_doc = lower_doc.apply(lambda x : re.sub(r'#\w+', '', x))
    
     # removing URL
    lower_doc = lower_doc.apply(lambda i : re.sub(r'https?://[^\s<>"]+|www\.[^\s<>"]+|http?://[^\s<>"]+','', i))
    # remove HTML tags
    lower_doc = lower_doc.apply(lambda x: strip_html(x))
    
    # Handle punctuation within words
    lower_doc = lower_doc.apply(lambda x : cont_to_exp(x))
    
    # remove no-alphanumeric
    lower_doc = lower_doc.apply(lambda x : re.sub(r'[^\w\s]', '', x))
    
    # remove stopwords
    lower_doc = lower_doc.apply(lambda x : " ".join(t for t in x.split() if t not in STOP_WORDS))
    
    # change text to base words
    # lower_doc = lower_doc.apply(lambda x : base_word(x))
    
    
    return lower_doc

df['processed_text'] = preprocess(df['text'])
full_text = " ".join(df['processed_text'])
doc = nlp(full_text)

# print(doc.type())




# print(nlp)
# def get_text():
#     doc = pd.read_csv('Tweets.csv')
    
#     return doc['text']

def main():

    orig_stdout = sys.stdout
    f = open('out.txt', 'w')
    sys.stdout = f
    
    for sen in df['processed_text'] :
        print(sen + "\n")
    
    sys.stdout = orig_stdout
    f.close()
    
    
if __name__ == "__main__":
    main()