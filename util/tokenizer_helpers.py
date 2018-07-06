# Keras Tokenizer Serialization
# Written by Gautam Mittal

import pickle

def save_tokenizer(filepath, tokenizer):
    with open(filepath, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_tokenizer(filepath):
    with open(filepath, 'rb') as handle:
        return pickle.load(handle)

def save_word_index(filepath, tokenizer):
    handle = open(filepath, 'rb')
    handle.write(str(tokenizer.word_index))
    handle.close()
