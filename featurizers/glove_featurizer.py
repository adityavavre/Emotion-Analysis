from typing import List
import os
import numpy as np


class GloveFeaturizer():
    def __init__(self, embedding_dim: int = 50):
        self.embeddings_dict = {}
        self.embedding_dim = embedding_dim
        print("Using GloVe embedding dim: ", embedding_dim)

        embedding_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                      "data/glove/glove.6B.{dim}d.txt".format(dim=self.embedding_dim))
        print("Reading GloVe embedding file: ", embedding_file)
        with open(embedding_file, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                self.embeddings_dict[word] = vector

    def featurize(self, sentences: List, tokenizer):
        features = []
        # print("Collecting avg. sentence embeddings from GloVe")
        for sent in sentences:
            tokenize_word = tokenizer.tokenize(sent)
            vec = np.array([0.] * self.embedding_dim, dtype='float32')
            count = 0
            for word in tokenize_word:
                try:
                    vec += self.embeddings_dict.get(word)
                    count += 1
                except:
                    continue

            if count != 0:
                vec /= count
            features.append(vec)
        return features
