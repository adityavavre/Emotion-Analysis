import pickle
from typing import List, Dict

import pandas as pd
import numpy as np

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
from utils import preprocess, read_data_from_dir, read_meld_data


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

    def featurize(self, sentences, tokenizer):
        features = {}
        for name, featurizer in self.featurizers.items():
            print("Starting %s feature extraction" % name)
            temp = featurizer.featurize(sentences, tokenizer)
            if isinstance(temp, Dict):
                features.update(temp)
            else:
                features[name] = temp

        return features

def extract_features(sentences: List, emotions: List, corpus: List, out_file: str):
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

    features = featurizer.featurize(pre_processed_sentences, tokenizer=tokenizer)

    features["emotions"] = emotions

    save_features(features, out_file)

    return features

def save_features(features: Dict, out_file: str):
    data = None
    col_names = []
    for name, vec in features.items():
        v = np.array(vec)
        if len(v.shape) == 1:
            v = v.reshape((-1, 1))
        for i in range(v.shape[1]):
            col_names.append(name + '_' + str(i))
        if data is not None:
            data = np.concatenate((data, v), axis=1)
        else:
            data = v
    print("Saving features with shape: ", data.shape)

    df = pd.DataFrame(data, columns=col_names, index=None)
    print("Saving dataframe to: ", out_file)
    df.to_csv(out_file, index=False)
    print("Saved")

if __name__ == '__main__':
    train_utterances, train_emotions, _ = read_data_from_dir('./data/dailydialog', split="train")
    valid_utterances, valid_emotions, _ = read_data_from_dir('./data/dailydialog', split="validation")
    test_utterances, test_emotions, _ = read_data_from_dir('./data/dailydialog', split="test")

    _ = extract_features(train_utterances, emotions=train_emotions, corpus=train_utterances,
                                      out_file='./data/dailydialog/train_features.csv')
    _ = extract_features(valid_utterances, emotions=valid_emotions, corpus=train_utterances,
                                      out_file='./data/dailydialog/validation_features.csv')
    _ = extract_features(test_utterances, emotions=test_emotions, corpus=train_utterances,
                                     out_file='./data/dailydialog/test_features.csv')

    meld_train_utterances, meld_train_emotions = read_meld_data('./data/meld', split="train")
    meld_valid_utterances, meld_valid_emotions = read_meld_data('./data/meld', split="dev")
    meld_test_utterances, meld_test_emotions = read_meld_data('./data/meld', split="test")

    train_utterances.extend(meld_train_utterances)
    train_emotions.extend(meld_train_emotions)
    valid_utterances.extend(meld_valid_utterances)
    valid_emotions.extend(meld_valid_emotions)
    test_utterances.extend(meld_test_utterances)
    test_emotions.extend(meld_test_emotions)

    _ = extract_features(train_utterances, emotions=train_emotions, corpus=train_utterances,
                                      out_file='./data/dailydialog/train_features_combined.csv')
    _ = extract_features(valid_utterances, emotions=valid_emotions, corpus=train_utterances,
                                      out_file='./data/dailydialog/validation_features_combined.csv')
    _ = extract_features(test_utterances, emotions=test_emotions, corpus=train_utterances,
                                     out_file='./data/dailydialog/test_features_combined.csv')

    # with open('./data/dailydialog/train_features.pkl', 'wb') as f:
    #     pickle.dump(train_features, f)
    #
    # with open('./data/dailydialog/validation_features.pkl', 'wb') as f:
    #     pickle.dump(valid_features, f)
    #
    # with open('./data/dailydialog/test_features.pkl', 'wb') as f:
    #     pickle.dump(test_features, f)
