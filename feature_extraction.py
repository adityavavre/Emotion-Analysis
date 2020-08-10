from typing import List
import numpy as np
from nltk.tokenize import word_tokenize

def get_glove(sentences: List):
    embeddings_dict = {}
    embedding_dim = 50
    with open("glove.6B.50d.txt", 'r') as f:
        for line in f:
            values = line.split()
            print(values)
            exit(0)
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings_dict[word] = vector

    # all_words = []
    feature = []
    for sent in sentences:
        tokenize_word = word_tokenize(sent)
        vec = np.array([0]*embedding_dim, dtype='float32')
        for word in tokenize_word:
            # all_words.append(word)
            vec += embeddings_dict.get(word)
        vec /= len(tokenize_word)
        feature.append(vec)
    return feature

def extract_features(sentences: List):
    features = []


    return features
