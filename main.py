import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.preprocessing.text import text_to_word_sequence

# Load the training data
data = pd.read_csv('data/train.csv')
data.fillna(' ')
data['full_text'] = data['title'] + ' ' + data['author'] + ' ' + data['text']
print(data['full_text'][0])
