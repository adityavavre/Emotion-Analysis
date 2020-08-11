import pickle
from typing import List, Dict
import numpy as np
from nltk.tokenize import word_tokenize
from featurizers.glove_featurizer import GloveFeaturizer
from featurizers.tf_idf_featurizer import TfIdfFeaturizer
from featurizers.ngram_featurizer import NgramFeaturizer
from featurizers.pos_count_featurizer import POSCountFeaturizer

from utils import preprocess, read_data_from_dir


class DailyDialogFeaturizer():
    def __init__(self, corpus: List):
        self.featurizers = {
            "GloVe": GloveFeaturizer(embedding_dim=100),
            "TF-IDF": TfIdfFeaturizer(corpus),
            "N-gram": NgramFeaturizer(corpus),
            "POScounts": POSCountFeaturizer(),
            # "AFINNValence": AFINNValenceFeaturizer(),
            # "BingLiu": BingLiuFeaturizer(),
            # "MPQAEffect": MPQAEffectFeaturizer(),
            # "NRCAffect": NRCAffectIntensityFeaturizer(),
            # "NRCEmotion": NRCEmotionFeaturizer(),
            # "NRCExpandedEmotion": NRCExpandedEmotionFeaturizer(),
            # "Sentiment140": Sentiment140Featurizer(),
            # "SentiWordNet": SentiWordNetFeaturizer(),
            # "SentiStrength": SentiStrengthFeaturizer(),
            # "Negation": NegationFeaturizer(),
            # "EdinburghEmbeddings": EdinburghEmbeddingsFeaturizer(),
            # "LIWC": LIWCFeaturizer()
        }

    def featurize(self, text, tokenizer):
        features = {}
        for name, featurizer in self.featurizers.items():
            temp = featurizer.featurize(text, tokenizer)
            if isinstance(temp, Dict):
                features.update(temp)
            else:
                features[name] = temp

        return features

def extract_features(sentences: List, corpus: List):
    # pre-process the corpus
    print("Pre-processing the corpus")
    pre_processed_corpus = list(map(lambda x: preprocess(x), corpus))
    print("Pre-processing done")

    # pre-process the utterances
    print("Pre-processing the utterances")
    pre_processed_sentences = list(map(lambda x: preprocess(x), sentences))
    print("Pre-processing done")

    featurizer = DailyDialogFeaturizer(pre_processed_corpus)

    return featurizer.featurize(pre_processed_sentences, tokenizer=word_tokenize)

if __name__ == '__main__':
    corpus, _, _ = read_data_from_dir('./data/dailydialog', split="train")
    utterances, emotions, _ = read_data_from_dir('./data/dailydialog', split="test")
    features = extract_features(utterances, corpus)

    with open('./data/dailydialog/features.pkl', 'wb') as f:
        pickle.dump(features, f)
