from featurizers.base_featurizers import SentimentIntensityLexiconFeaturizer
from featurizers.utils import get_bigrams, merge_two_dicts
from featurizers.utils import afinn_lexicon_path, afinn_emoticon_path

"""
Info: http://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=6010
"""


class AFINNValenceFeaturizer(SentimentIntensityLexiconFeaturizer):
    """AFINN Valence Lexicons Featurizer"""

    @property
    def id(self):
        return self._id

    @property
    def lexicon_map(self):
        return self._lexicon_map

    def __init__(self, lexicons_path=afinn_lexicon_path, emoticon_path=afinn_emoticon_path, bigram=True):
        """
        Initialize Finn Årup Nielsen Valence Lexicon Featurizer
        :param lexicons_path path to lexicons file
        :param emoticon_path path to emoticons file
        :param bigram use bigram lexicons or not (default: True)
        """
        self._id = 'AFINN'
        self._lexicon_map = merge_two_dicts(self.create_lexicon_mapping(lexicons_path),
                                 self.create_lexicon_mapping(emoticon_path))
        self.bigram = bigram

    def featurize(self, sentences, tokenizer):
        """
        Featurize tokens using AFINN Valence Lexicons
        :param text: lis of sentences to featurize
        :param tokenizer: tokenizer to tokenize text
        """
        features = []
        for text in sentences:
            tokens = tokenizer.tokenize(text)
            unigrams = tokens
            if self.bigram:
                bigrams = get_bigrams(tokens)
                features.append([x + y for x, y in zip(super(AFINNValenceFeaturizer, self).featurize_tokens(unigrams),
                                              super(AFINNValenceFeaturizer, self).featurize_tokens(bigrams))])

            else:
                features.append(super(AFINNValenceFeaturizer, self).featurize_tokens(unigrams))

        return features
