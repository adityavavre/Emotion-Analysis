# coding=utf-8
from featurizers.base_featurizers import EmotionLexiconFeaturizer
from featurizers.utils import nrc_expanded_emotion_lexicon_path

"""
Info: http://www.cs.waikato.ac.nz/ml/sa/lex.html
"""


class NRCExpandedEmotionFeaturizer(EmotionLexiconFeaturizer):
    """
    NRC Expanded Emotion Lexicon Featurizer
    """

    @property
    def id(self):
        return self._id

    @property
    def lexicon_map(self):
        return self._lexicon_map

    def __init__(self, lexicon_path=nrc_expanded_emotion_lexicon_path):
        """
        Initialize Bravo-Marquez NRC Expanded Emotion Lexicon featurizer
        :param lexicon_path path to unigram lexicons file
        """
        self._id = 'NRCExpandedEmotion'
        self._lexicon_map = self.create_lexicon_mapping(lexicon_path)
