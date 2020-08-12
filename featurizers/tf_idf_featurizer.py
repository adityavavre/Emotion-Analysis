from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer


class TfIdfFeaturizer():
    def __init__(self, corpus: List):
        self.vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            max_df=0.8,
            stop_words='english',
            analyzer='word',
            token_pattern=r'\w{1,}',
            ngram_range=(2,3),
            max_features=100)

        self.corpus = corpus

        print("Fitting corpus to tf-idf vectorizer")
        self.vectorizer.fit(self.corpus)
        print("Fit done")

    def featurize(self, sentences: List, tokenizer):
        print("Extracting tf-idf vector feature")
        return self.vectorizer.transform(sentences).toarray()
