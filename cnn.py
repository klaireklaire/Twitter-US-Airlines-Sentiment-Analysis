import tensorflow as tf
from nltk.tokenize import word_tokenize
import pandas as pd
import math
import numpy as np
from string import punctuation
from os import listdir
from numpy import array
from numpy import asarray
from numpy import zeros
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding, Conv1D, MaxPooling1D, Dropout
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns

def read_and_process_data():
    doc = pd.read_csv('cleaned_readable.csv')
    processed_text = doc["processed_text"]
    sentiments = doc["airline_sentiment"]

    processed_data = [(tweet, sentiment) for tweet, sentiment in zip(processed_text, sentiments) if isinstance(tweet, str)]


    tokens = [(word_tokenize(str(data[0])), data[1]) for data in processed_data]
    words = [item for row in tokens for item in row[0]]

    positive_tweets = [tweet for tweet, sentiment in tokens if sentiment == "positive"]
    negative_tweets = [tweet for tweet, sentiment in tokens if sentiment == "negative"]
    return positive_tweets, negative_tweets, words

# save list to file
def save_list(lines, filename):
	# convert lines to a single blob of text
	data = '\n'.join(lines)
	# open file
	file = open(filename, 'w')
	# write text
	file.write(data)
	# close file
	file.close()

# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


def load_embedding(filename):
    # load embedding into memory, skip first line
    file = open(filename,'r')
    lines = file.readlines()
    file.close()
    # create a map of words to vectors
    embedding = dict()
    for line in lines:
        parts = line.split()
        # key is string word, value is numpy array for vector
        embedding[parts[0]] = asarray(parts[1:], dtype='float32')
    return embedding

# create a weight matrix for the Embedding layer from a loaded embedding
def get_weight_matrix(embedding, vocab):
    # total vocabulary size plus 0 for unknown words
    vocab_size = len(vocab) + 1
    # define weight matrix dimensions with all 0
    weight_matrix = zeros((vocab_size, 100))
    # step vocab, store vectors using the Tokenizer's integer mapping
    for word, i in vocab.items():
        vector = embedding.get(word)
        if vector is not None:
            weight_matrix[i] = vector
    return weight_matrix

# load the vocabulary
def load_vocab():
    vocab_filename = 'vocab.txt'
    vocab = load_doc(vocab_filename)
    vocab = vocab.split()
    vocab = set(vocab)
    return vocab

# load all training reviews
def load_test_train_docs(positive_tweets, negative_tweets):
    split_index = math.floor(len(positive_tweets) * 0.7)
    train_positive_docs = positive_tweets[:split_index]
    test_positive_docs = positive_tweets[split_index:]

    split_index = math.floor(len(negative_tweets) * 0.7)
    train_negative_docs = negative_tweets[:split_index]
    test_negative_docs = negative_tweets[split_index:]
    return train_positive_docs, test_positive_docs, train_negative_docs, test_negative_docs

def tokenize_and_encode(train_positive_docs, test_positive_docs, train_negative_docs, test_negative_docs):
    train_docs = train_negative_docs + train_positive_docs
    tokenizer = Tokenizer()
    # fit the tokenizer on the documents
    tokenizer.fit_on_texts(train_docs)
    # sequence encode
    encoded_docs = tokenizer.texts_to_sequences(train_docs)
    # pad sequences
    max_length = max([len(s) for s in train_docs])
    Xtrain = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    # define training labels
    ytrain = array([0 for _ in range(6412)] + [1 for _ in range(1648)])
    
    # load all test reviews
    test_docs = test_negative_docs + test_positive_docs

    # sequence encode
    encoded_docs = tokenizer.texts_to_sequences(test_docs)
    # pad sequences
    Xtest = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    # define test labels

    ytest = array([0 for _ in range(2749)] + [1 for _ in range(707)])
    
    # define vocabulary size (largest integer value)
    vocab_size = len(tokenizer.word_index) + 1
    return tokenizer, Xtrain, ytrain, Xtest, ytest, vocab_size, max_length
    
def load_embed(tokenizer, vocab_size, max_length):
    # load embedding from file
    raw_embedding = load_embedding('glove.twitter.27B.100d.txt')
    # get vectors in the right order
    embedding_vectors = get_weight_matrix(raw_embedding, tokenizer.word_index)
    # create the embedding layer
    embedding_layer = Embedding(vocab_size, 100, weights=[embedding_vectors], input_length=max_length, trainable=False)
    return embedding_layer
 
def define_and_evaluate_model(embedding_layer, Xtrain, ytrain, Xtest, ytest):
    model = Sequential()
    model.add(embedding_layer)
    model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    # compile network
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    
    # fit network
    model.fit(Xtrain, ytrain, epochs=10, verbose=2)
    print(model.summary())
    
    y_pred = (model.predict(Xtest) > 0.5).astype("int32")
    
    print("Confusion Matrix:")
    cm = confusion_matrix(ytest, y_pred)
    print(classification_report(ytest, y_pred, target_names=['Negative', 'Positive']))
    total_correct = cm.diagonal()
    total_predictions = cm.sum(axis=1)
    accuracy_per_class = total_correct / total_predictions
    print(f"Accuracy for Negative: {accuracy_per_class[0]}")
    print(f"Accuracy for Positive: {accuracy_per_class[1]}")
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()
    
    return model

def main():
    positive_tweets, negative_tweets, words = read_and_process_data()
    # save tokens to a vocabulary file
    save_list(words, 'vocab.txt')
    train_positive_docs, test_positive_docs, train_negative_docs, test_negative_docs = load_test_train_docs(positive_tweets, negative_tweets)
    tokenizer, Xtrain, ytrain, Xtest, ytest, vocab_size, max_length = tokenize_and_encode(train_positive_docs, test_positive_docs, train_negative_docs, test_negative_docs)
    embedding_layer = load_embed(tokenizer, vocab_size, max_length)
    model = define_and_evaluate_model(embedding_layer, Xtrain, ytrain, Xtest, ytest)
  
    
    
if __name__ == "__main__":
    main()