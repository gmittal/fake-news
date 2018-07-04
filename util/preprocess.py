import re
STOP = open('util/stopwords.txt').read().split('\n')[:-1]

def preprocess(text):
    text = text.lower() # Lowercase
    text = re.sub(r'[^\w\s]','',text, re.UNICODE) # Remove punctuation
    text = " ".join(x for x in text.split() if x not in STOP) # Remove stopwords
    return text
