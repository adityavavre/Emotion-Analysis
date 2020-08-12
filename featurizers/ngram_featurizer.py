from typing import List

from sklearn.feature_extraction.text import CountVectorizer


class NgramFeaturizer():
    def __init__(self, corpus: List):
        self.vectorizer =CountVectorizer(
            ngram_range=(2, 3),
            stop_words='english',
            analyzer='word',
            token_pattern=r'\b\w+\b',
            max_features=100)

        self.corpus = corpus

        print("Fitting corpus to n-gram(n=2,3) vectorizer")
        self.vectorizer.fit(self.corpus)
        print("Fit done")

    def featurize(self, sentences: List, tokenizer):
        # print("Extracting word n-grams vector feature")
        return self.vectorizer.transform(sentences).toarray()
