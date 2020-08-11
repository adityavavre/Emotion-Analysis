import pickle
from typing import List, Dict

from featurizers.afinn_valence_featurizer import AFINNValenceFeaturizer
from featurizers.bing_liu_sentiment_featurizer import BingLiuFeaturizer
from featurizers.edinburgh_embeddings_featurizer import EdinburghEmbeddingsFeaturizer
from featurizers.glove_featurizer import GloveFeaturizer
from featurizers.liwc_featurizer import LIWCFeaturizer
from featurizers.mpqa_effect_featurizer import MPQAEffectFeaturizer
from featurizers.negating_featurizer import NegationFeaturizer
from featurizers.ngram_featurizer import NgramFeaturizer
from featurizers.nrc_affect_intensity_featurizer import NRCAffectIntensityFeaturizer
from featurizers.nrc_emotion_wordlevel_featurizer import NRCEmotionFeaturizer
from featurizers.nrc_expanded_emotion_featurizer import NRCExpandedEmotionFeaturizer
from featurizers.pos_count_featurizer import POSCountFeaturizer
from featurizers.senti_wordnet_featurizer import SentiWordNetFeaturizer
from featurizers.sentiment140_featurizer import Sentiment140Featurizer
from featurizers.tf_idf_featurizer import TfIdfFeaturizer
from utils import Tokenizer
from utils import preprocess, read_data_from_dir


class DailyDialogFeaturizer():
    def __init__(self, corpus: List):
        self.featurizers = {
            "GloVe": GloveFeaturizer(embedding_dim=100),
            "TF-IDF": TfIdfFeaturizer(corpus),
            "N-gram": NgramFeaturizer(corpus),
            "POScounts": POSCountFeaturizer(),
            "AFINNValence": AFINNValenceFeaturizer(),
            "BingLiu": BingLiuFeaturizer(),
            "MPQAEffect": MPQAEffectFeaturizer(),
            "NRCAffect": NRCAffectIntensityFeaturizer(),
            "NRCEmotion": NRCEmotionFeaturizer(),
            "NRCExpandedEmotion": NRCExpandedEmotionFeaturizer(),
            "Sentiment140": Sentiment140Featurizer(),
            "SentiWordNet": SentiWordNetFeaturizer(),
            "Negation": NegationFeaturizer(),
            "EdinburghEmbeddings": EdinburghEmbeddingsFeaturizer(),
            "LIWC": LIWCFeaturizer()
        }

    def featurize(self, text, tokenizer):
        features = {}
        for name, featurizer in self.featurizers.items():
            print("Starting %s feature extraction" % name)
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
    tokenizer = Tokenizer()

    return featurizer.featurize(pre_processed_sentences, tokenizer=tokenizer)

if __name__ == '__main__':
    corpus, _, _ = read_data_from_dir('./data/dailydialog', split="train")
    utterances, emotions, _ = read_data_from_dir('./data/dailydialog', split="test")
    features = extract_features(utterances, corpus)

    with open('./data/dailydialog/features.pkl', 'wb') as f:
        pickle.dump(features, f)
