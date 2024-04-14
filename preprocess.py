#!/usr/bin/python3

import nltk
import pandas as pd
import sys
import re
from bs4 import BeautifulSoup


doc = pd.read_csv('Tweets.csv')
df = pd.DataFrame(doc)

def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")                    
    return soup.get_text()

def preprocess(doc):
    lower_doc = doc.str.lower()
    lower_doc = lower_doc.replace('[^\w\s]','')
    # removing URL
    lower_doc = lower_doc.apply(lambda i : re.sub(r'https?://[^\s<>"]+|www\.[^\s<>"]+','', i))
    # remove HTML tags
    lower_doc = lower_doc.apply(lambda x: strip_html(x))
    
    
    
    return lower_doc

df['processed_text'] = preprocess(df['text'])
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