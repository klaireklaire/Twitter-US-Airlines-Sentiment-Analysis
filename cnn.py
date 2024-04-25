import tensorflow as tf
# from tf import keras

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense

# model = Sequential([
#     Embedding(input_dim=10000, output_dim=128, input_length=100),  # Assuming vocab size of 10,000 and input length of 100
#     Conv1D(64, 5, activation='relu'),  # 64 filters and a kernel size of 5
#     GlobalMaxPooling1D(),
#     Dense(10, activation='relu'),
#     Dense(1, activation='sigmoid')  # Output layer for binary classification
# ])

# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences

# texts = ["This is a positive example.", "This is a negative example."]  # Example texts
# tokenizer = Tokenizer(num_words=10000)
# tokenizer.fit_on_texts(texts)
# sequences = tokenizer.texts_to_sequences(texts)
# data = pad_sequences(sequences, maxlen=100)

# import numpy as np

# labels = np.array([1, 0])  # Example binary labels for the texts
# model.fit(data, labels, epochs=10, validation_split=0.2)

# # Assume `test_data` and `test_labels` are prepared in the same way as `data` and `labels`
# test_loss, test_acc = model.evaluate(data, labels)
# print(f"Test Accuracy: {test_acc}")
from nltk.tokenize import word_tokenize
import pandas as pd

from string import punctuation
from os import listdir
from numpy import array
from numpy import asarray
from numpy import zeros
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Embedding
from sklearn.model_selection import train_test_split
# from tensorflow.keras.layers.convolutional import Conv1D
# from tensorflow.keras.layers.convolutional import MaxPooling1D

# doc = pd.read_csv('cleaned_readable.csv')
# processed_text = doc["processed_text"]
# tokens = [word_tokenize(str(sent)) for sent in processed_text]
# tokens = [item for row in tokens for item in row]

# # save list to file
# def save_list(lines, filename):
# 	# convert lines to a single blob of text
# 	data = '\n'.join(lines)
# 	# open file
# 	file = open(filename, 'w')
# 	# write text
# 	file.write(data)
# 	# close file
# 	file.close()

# # save tokens to a vocabulary file
# save_list(tokens, 'vocab.txt')

# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

# load all docs in a directory
# def process_docs(directory, vocab, is_trian):
#     documents = list()
#     # walk through all files in the folder
#     for filename in listdir(directory):
#         # skip any reviews in the test set
#         if is_trian and filename.startswith('cv9'):
#             continue
#         if not is_trian and not filename.startswith('cv9'):
#             continue
#         # create the full path of the file to open
#         path = directory + '/' + filename
#         # load the doc
#         doc = load_doc(path)
#         # clean doc
#         tokens = clean_doc(doc, vocab)
#         # add to list
#         documents.append(tokens)
#     return documents

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
vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)




# load all training reviews
positive_docs = process_docs('txt_sentoken/pos', vocab, True)
negative_docs = process_docs('txt_sentoken/neg', vocab, True)
train_docs = negative_docs + positive_docs

xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=42)

# create the tokenizer
tokenizer = Tokenizer()
# fit the tokenizer on the documents
tokenizer.fit_on_texts(train_docs)

# sequence encode
encoded_docs = tokenizer.texts_to_sequences(train_docs)
# pad sequences
max_length = max([len(s.split()) for s in train_docs])
Xtrain = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# define training labels
ytrain = array([0 for _ in range(900)] + [1 for _ in range(900)])
 
# load all test reviews
positive_docs = process_docs('txt_sentoken/pos', vocab, False)
negative_docs = process_docs('txt_sentoken/neg', vocab, False)
test_docs = negative_docs + positive_docs
# sequence encode
encoded_docs = tokenizer.texts_to_sequences(test_docs)
# pad sequences
Xtest = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# define test labels
ytest = array([0 for _ in range(100)] + [1 for _ in range(100)])
 
# define vocabulary size (largest integer value)
vocab_size = len(tokenizer.word_index) + 1

# load embedding from file
raw_embedding = load_embedding('glove.twitter.27B.100d.txt')
# get vectors in the right order
embedding_vectors = get_weight_matrix(raw_embedding, tokenizer.word_index)
# create the embedding layer
embedding_layer = Embedding(vocab_size, 100, weights=[embedding_vectors], input_length=max_length, trainable=False)
 
# define model
model = Sequential()
model.add(embedding_layer)
model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
print(model.summary())
# compile network
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
model.fit(Xtrain, ytrain, epochs=10, verbose=2)
# evaluate
loss, acc = model.evaluate(Xtest, ytest, verbose=0)
print('Test Accuracy: %f' % (acc*100))
