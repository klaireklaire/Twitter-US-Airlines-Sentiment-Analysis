import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense

model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=100),  # Assuming vocab size of 10,000 and input length of 100
    Conv1D(64, 5, activation='relu'),  # 64 filters and a kernel size of 5
    GlobalMaxPooling1D(),
    Dense(10, activation='relu'),
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

texts = ["This is a positive example.", "This is a negative example."]  # Example texts
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
data = pad_sequences(sequences, maxlen=100)

import numpy as np

labels = np.array([1, 0])  # Example binary labels for the texts
model.fit(data, labels, epochs=10, validation_split=0.2)

# Assume `test_data` and `test_labels` are prepared in the same way as `data` and `labels`
test_loss, test_acc = model.evaluate(data, labels)
print(f"Test Accuracy: {test_acc}")
