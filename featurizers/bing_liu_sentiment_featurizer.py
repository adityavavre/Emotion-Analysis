from featurizers.base_featurizers import SentimentLexiconFeaturizer
from featurizers.utils import bing_liu_lexicon_path

"""
Info: https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html#lexicon
"""


class BingLiuFeaturizer(SentimentLexiconFeaturizer):
    """
    Bing Liu Sentiment Lexicon Featurizer
    """

    @property
    def id(self):
        return self._id

    @property
    def lexicon_map(self):
        return self._lexicon_map


    def __init__(self, lexicons_path=bing_liu_lexicon_path):
        """Initialize BingLiu Sentiment Lexicon Featurizer
        :param lexicons_path path to lexicons file
        """
        self._id = 'BingLiu'
        self._lexicon_map = self.create_lexicon_mapping(lexicons_path)
