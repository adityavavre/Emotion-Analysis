from featurizers.base_featurizers import EmotionLexiconFeaturizer
from featurizers.utils import nrc_emotion_lexicon_path

"""
Info: http://saifmohammad.com/WebPages/lexicons.html
"""


class NRCEmotionFeaturizer(EmotionLexiconFeaturizer):
    """
    NRC Wordlevel Emotion Lexicon Featurizer
    """

    @property
    def id(self):
        return self._id

    @property
    def lexicon_map(self):
        return self._lexicon_map

    def __init__(self, lexicon_path=nrc_emotion_lexicon_path):
        """
        Initialize Saif Mohammad NRC Wordlevel Emotion Lexicon featurizer
        :param lexicon_path path to unigram lexicons file
        """
        self._id = 'NRCEmotionWordlevel'
        self._lexicon_map = self.create_lexicon_mapping(lexicon_path)
