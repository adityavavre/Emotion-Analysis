from featurizers.base_featurizers import Featurizer
from featurizers.utils import negation_lexicon_path
import gzip
from collections import defaultdict


class NegationFeaturizer(Featurizer):
    """Negation Featurizer"""

    @property
    def id(self):
        return self._id

    @property
    def lexicon_map(self):
        return self._lexicon_map

    @property
    def features(self):
        return [self.id + "-" + "count"]

    @staticmethod
    def create_lexicon_mapping(lexicon_path):
        """
        Creates a map from lexicons to either positive or negative
        :param lexicon_path path of lexicon file (in gzip format)
        """
        with gzip.open(lexicon_path, 'rb') as f:
            lines = f.read().splitlines()
            lexicon_map = defaultdict(int)
            for l in lines:
                splits = l.decode('utf-8').split('\t')
                lexicon_map[splits[0]] += 1
        return lexicon_map

    def __init__(self, lexicons_path=negation_lexicon_path):
        """Initialize Negation Count Featurizer
        :param lexicons_path path to lexicons file
        """
        self._id = 'Negation'
        self._lexicon_map = self.create_lexicon_mapping(lexicons_path)

    def featurize(self, sentences, tokenizer):
        """

        :param sentences:
        :param tokenizer:
        :return:
        """
        features= []
        for text in sentences:
            count = 0
            tokens = tokenizer.tokenize(text)
            for token in tokens:
                if token in self.lexicon_map:
                    count += 1
            features.append([count])
        return features
