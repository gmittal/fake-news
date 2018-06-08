import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()

# Load the training data
data = pd.read_csv('data/train.csv')
data = data.fillna(' ')
# data['full_text'] = data['title'] + ' ' + data['author'] + ' ' + data['text']
# tokenizer.fit_on_texts(data['full_text'].values)
# train_docs = tokenizer.texts_to_matrix(data['full_text'].values, mode='tfidf')
train_labels = data['label'].values #tf.keras.utils.to_categorical(data['label'].values, 2)
# np.save('data/train_mat.npy', train_docs)
train_docs = np.load('data/train_mat.npy')

# Load testing data
# test = pd.read_csv('data/test.csv')
# test = test.fillna(' ')
# test['full_text'] = test['title'] + ' ' + test['author'] + ' ' + test['text']
# test_docs = tokenizer.texts_to_matrix(test['full_text'].values, mode='tfidf')
# np.save('data/test_mat.npy', test_docs)
#
# print(train_docs.shape)
# print(test_docs.shape)

model = Sequential()
model.add(Dense(10000, input_shape=(242483,), activation='relu'))
model.add(Dense(1, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.fit(train_docs, train_labels, validation_split=0.2, epochs=10, batch_size=128)
