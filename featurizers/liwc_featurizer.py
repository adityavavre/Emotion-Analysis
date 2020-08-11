import re

import nltk
import numpy as np
from collections import Counter

from featurizers.base_featurizers import Featurizer
from .utils import liwc_lexicon_path

"""
Info: http://liwc.wpengine.com/
"""

class LIWCTrie:
    def __init__(self):
        self.root = dict()
        self.end = '$'
        self.cont = '*'

    def insert(self, word, categories):
        """
        Insert liwc word and categories in trie
        :param word: word to insert
        :param categories: liwc categories
        :return: None
        """
        if len(word) == 0:
            return
        curr_dict = self.root
        for letter in word[:-1]:
            curr_dict = curr_dict.setdefault(letter, {})
        if word[-1] != self.cont:
            curr_dict = curr_dict.setdefault(word[-1], {})
            curr_dict[self.end] = categories
        else:
            curr_dict[self.cont] = categories

    def in_trie(self, word):
        """
        Query if a word is in trie or not
        :param word: query word
        :return: True if word is present otherwise False
        """
        curr_dict = self.root
        for letter in word:
            if letter in curr_dict:
                curr_dict = curr_dict[letter]
            elif self.cont in curr_dict:
                return True
            else:
                return False
        if self.cont in curr_dict or self.end in curr_dict:
            return True
        else:
            return False

    def get(self, word):
        """
        get value stored against word
        :param word: word to search
        :return: list of categories if word is in query otherwise None
        """
        curr_dict = self.root
        for letter in word:
            if letter in curr_dict:
                curr_dict = curr_dict[letter]
            elif self.cont in curr_dict:
                return curr_dict[self.cont]
            else:
                return None
        if self.cont in curr_dict or self.end in curr_dict:
            if self.cont in curr_dict:
                return curr_dict[self.cont]
            else:
                return curr_dict[self.end]
        else:
            return None

    def __getitem__(self, item):
        return self.get(item)

    def __contains__(self, item):
        return self.in_trie(item)

class LIWCFeaturizer(Featurizer):
    """
    LIWC Lexicon Featurizer
    """

    @property
    def features(self):
        return self._features

    @property
    def id(self):
        return self._id

    @staticmethod
    def create_lexicon_mapping(lexicon_path):
        liwc_trie = LIWCTrie()
        data = open(lexicon_path).read()
        splits = data.split('%')
        categories = dict([(x.split('\t')[0], x.split('\t')[1]) for x in splits[1].strip().splitlines()])
        for x in splits[2].strip().splitlines():
            try:
                pair = (x.split('\t')[0], [categories[y] for y in x.strip().split('\t')[1:]])
                liwc_trie.insert(pair[0], pair[1])
            except Exception as ex:
                pass
        return list(categories.values()), liwc_trie

    def __init__(self, lexicons_path=liwc_lexicon_path):
        """Initialize LIWC Lexicon Featurizer
        :param lexicons_path path to lexicons file
        """
        nltk.download('punkt', quiet=True)
        self._id = 'LIWC'
        self.categories, self.liwc_trie = self.create_lexicon_mapping(lexicons_path)
        self._features = ['total_word_count', 'avg_sentence_length', 'dictionary_words',
                          '>six_letter_words', 'numerals'] + self.categories + [x[0] for x in self.punctuations()]
        self._features.sort()
        self._citation = ''  # Todo

    @staticmethod
    def number_of_words(text):
        return len(re.findall(r"[a-z]['a-z]*", text.lower()))

    @staticmethod
    def percentage(a, b):
        # return a
        return (a * 100.0) / (b * 1.0 + 1.0)

    @staticmethod
    def punctuations():
        return [('Period', '.'), ('Comma', ','), ('Colon', ':'), ('SemiC', ';'), ('QMark', '?'), ('Exclam', '!'),
                ('Dash', '-'), ('Quote', '"'), ('Apostro', "'"), ('Parenth', '()[]{}'),
                ('OtherP', '#$%&*+-/<=>@\\^_`|~')]

    def set_punctuation_counts(self, text, liwc):
        character_counts, counts = Counter(text), {}
        for name, chars in self.punctuations():
            counts[name] = sum(character_counts[char] for char in chars)
        counts['Parenth'] /= 2.0
        counts['AllPct'] = sum(counts[name] for name, _ in self.punctuations())
        for x, y in counts.items():
            liwc[x] = self.percentage(y, liwc['total_word_count'])

    def featurize(self, sentences, tokenizer):
        features = []
        for text in sentences:
            liwc = {}
            if not isinstance(text, str):
                text = text.decode('utf8')

            num_capital_words = len(re.findall(r"[A-Z]['A-Z]*", text))
            words = re.findall(r"[a-z]['a-z]*", text.lower())

            text = text.lower()
            num_words = len(words)

            # text level features
            liwc['total_word_count'] = num_words
            liwc['num_capital_words'] = self.percentage(num_capital_words, num_words)
            if len(nltk.sent_tokenize(text)) > 0:
                liwc['avg_sentence_length'] = np.mean([self.number_of_words(sent)
                                                       for sent in nltk.sent_tokenize(text)])
            else:
                liwc['avg_sentence_length'] = 1.0
            liwc['>six_letter_words'] = self.percentage(sum([1 for x in words if len(x) >= 6]), num_words)
            liwc['dictionary_words'] = self.percentage(sum([1 for x in words if x in self.liwc_trie]), num_words)
            liwc['numerals'] = self.percentage(sum([1 for x in words if x.isdigit()]), num_words)

            for cat in self.categories:
                liwc[cat] = 0.0

            # categorical features
            tokens = tokenizer.tokenize(text)
            for token in tokens:
                if token in self.liwc_trie:
                    for cat in self.liwc_trie[token]:
                        liwc[cat] += 1.0

            for cat in self.categories:
                liwc[cat] = self.percentage(liwc[cat], num_words)

            self.set_punctuation_counts(text, liwc)
            features.append([liwc[x] for x in self.features])

        return features
