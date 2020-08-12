from typing import List

import textblob


class POSCountFeaturizer():
    def __init__(self):
        self.pos_family = {
            'noun': ['NN', 'NNS', 'NNP', 'NNPS'],
            'pron': ['PRP', 'PRP$', 'WP', 'WP$'],
            'verb': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
            'adj': ['JJ', 'JJR', 'JJS'],
            'adv': ['RB', 'RBR', 'RBS', 'WRB']
        }

    # function to check and get the part of speech tag count of a words in a given sentence
    def check_pos_tag(self, x, flag):
        cnt = 0
        try:
            wiki = textblob.TextBlob(x)
            for tup in wiki.tags:
                ppo = list(tup)[1]
                if ppo in self.pos_family[flag]:
                    cnt += 1
        except:
            pass
        return cnt

    def featurize(self, sentences: List, tokenizer):
        # print("Extracting POS counts features")
        counts = {}

        counts['noun_count'] = list(map(lambda x: self.check_pos_tag(x, 'noun'), sentences))
        counts['verb_count'] = list(map(lambda x: self.check_pos_tag(x, 'verb'), sentences))
        counts['adj_count'] = list(map(lambda x: self.check_pos_tag(x, 'adj'), sentences))
        counts['adv_count'] = list(map(lambda x: self.check_pos_tag(x, 'adv'), sentences))
        counts['pron_count'] = list(map(lambda x: self.check_pos_tag(x, 'pron'), sentences))

        return counts
