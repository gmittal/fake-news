import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Embedding
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Load the training and test data
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')
train_data = train_data.fillna(' ')
test_data = test_data.fillna(' ')
y = train_data['label'].values # Class values

train_sent = train_data['title'] + ' ' + train_data['author'] + ' ' + train_data['text']
test_sent = test_data['title'] + ' ' + test_data['author'] + ' ' + test_data['text']

tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(list(train_sent))
train_tokens = tokenizer.texts_to_sequences(train_sent)
test_tokens = tokenizer.texts_to_sequences(test_sent)

train = pad_sequences(train_tokens, maxlen=1000)
test = pad_sequences(test_tokens, maxlen=1000)

model = Sequential()
model.add(Embedding(20000, 128, input_length=1000))
model.add(Bidirectional(LSTM(60, return_sequences=True)))
model.add(GlobalMaxPool1D())
model.add(Dropout(0.1))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train, y, batch_size=32, epochs=2, validation_split=0.1)
