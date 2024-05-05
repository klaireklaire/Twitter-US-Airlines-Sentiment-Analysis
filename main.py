from preprocess import preprocess, vectorize, vectorizer
from ml import tune_train_evaluate_mnb_muticlass, get_top_features, plot_top_words, extract_words
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from lda import latent_dirichlet_allocation
import numpy as np
import sys

import pandas as pd

doc = pd.read_csv('Tweets.csv')
df = pd.DataFrame(doc)

def main():

    #df['processed_text'] = preprocess(df['text'])
    #full_text = " ".join(df['processed_text'])
    #doc = nlp(full_text)

    # df.to_pickle("cleaned_no_stem.csv")
    # df.to_csv("cleaned_readable.csv")
    df_cleaned = pd.read_pickle("cleaned_no_stem.csv")
    print(df_cleaned.head(10)['processed_text'])
    X = vectorize(df_cleaned['processed_text'], is_train=True)
    y = df_cleaned['airline_sentiment']

    # 70% for training, 30% for validation
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=42)

    #nb_best = tune_train_evaluate_mnb_muticlass(X, y, xtrain, xtest, ytrain, ytest)
    #joblib.dump(nb_best, 'mnb_model.pkl')

    nb_best = joblib.load('mnb_model.pkl')
    # feature_names = vectorizer.get_feature_names_out()
    # all_keywords = get_top_features(nb_best, feature_names)
    # positive = latent_dirichlet_allocation(all_keywords[2])
    # print(all_keywords[2])
    # print(positive)

    feature_names = vectorizer.get_feature_names_out()
    class_features = get_top_features(nb_best, feature_names, top_n=500)
    #plot_top_words(class_features)

    # LDA for positive class
    pos_words = extract_words(class_features, 'positive')
    print(pos_words)
    positive = latent_dirichlet_allocation(pos_words)

    # LDA for negative class
    neg_words = extract_words(class_features, 'negative')
    print(neg_words)
    negative = latent_dirichlet_allocation(neg_words)   
    

    orig_stdout = sys.stdout
    f = open('out.txt', 'w')
    sys.stdout = f


    
    for sen in df_cleaned['processed_text'] :
        print(sen + "\n")
    
    sys.stdout = orig_stdout
    f.close()
    
    
if __name__ == "__main__":
    main()