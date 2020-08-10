import pickle
from typing import List
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import textblob

from utils import preprocess, read_data_from_dir


def get_glove(sentences: List):
    embeddings_dict = {}
    embedding_dim = 50
    print("Using embedding dim: ", embedding_dim)
    print("Reading GloVe embedding file")
    with open("./data/glove/glove.6B.50d.txt", 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings_dict[word] = vector

    feature = []
    print("Collecting avg. sentence embeddings from GloVe")
    for sent in sentences:
        tokenize_word = word_tokenize(sent)
        vec = np.array([0.]*embedding_dim, dtype='float32')
        count = 0
        for word in tokenize_word:
            try:
                vec += embeddings_dict.get(word)
                count += 1
            except:
                continue

        if count != 0:
            vec /= count
        feature.append(vec)
    return feature

def get_tf_idf(sentences: List, corpus: List):
    vectorizer = TfidfVectorizer(sublinear_tf=True,
                                 max_df=0.8,
                                 stop_words='english',
                                 analyzer='word',
                                 token_pattern=r'\w{1,}',
                                 ngram_range=(2,3),
                                 max_features=5000)
    print("Fitting corpus to tf-idf vectorizer")
    vectorizer.fit(corpus)
    print("Fit done")

    return vectorizer.transform(sentences).toarray()

def get_word_ngrams(sentences: List, corpus: List):
    vectorizer = CountVectorizer(ngram_range=(2, 3),
                                 stop_words='english',
                                 analyzer='word',
                                 token_pattern=r'\b\w+\b',
                                 min_df=1,
                                 max_features=100)
    print("Fitting corpus to n-gram(n=2,3) vectorizer")
    vectorizer.fit(corpus)
    print("Fit done")

    return vectorizer.transform(sentences).toarray()


def get_pos_counts(sentences: List):
    counts = {}
    pos_family = {
        'noun': ['NN', 'NNS', 'NNP', 'NNPS'],
        'pron': ['PRP', 'PRP$', 'WP', 'WP$'],
        'verb': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
        'adj': ['JJ', 'JJR', 'JJS'],
        'adv': ['RB', 'RBR', 'RBS', 'WRB']
    }

    # function to check and get the part of speech tag count of a words in a given sentence
    def check_pos_tag(x, flag):
        cnt = 0
        try:
            wiki = textblob.TextBlob(x)
            for tup in wiki.tags:
                ppo = list(tup)[1]
                if ppo in pos_family[flag]:
                    cnt += 1
        except:
            pass
        return cnt

    counts['noun_count'] = list(map(lambda x: check_pos_tag(x, 'noun'), sentences))
    counts['verb_count'] = list(map(lambda x: check_pos_tag(x, 'verb'), sentences))
    counts['adj_count'] = list(map(lambda x: check_pos_tag(x, 'adj'), sentences))
    counts['adv_count'] = list(map(lambda x: check_pos_tag(x, 'adv'), sentences))
    counts['pron_count'] = list(map(lambda x: check_pos_tag(x, 'pron'), sentences))

    return counts

def extract_features(sentences: List, corpus: List):
    features = {}

    # pre-process the corpus
    print("Pre-processing the corpus")
    pre_processed_corpus = list(map(lambda x: preprocess(x), corpus))
    print("Pre-processing done")

    # pre-process the utterances
    print("Pre-processing the utterances")
    pre_processed_sentences = list(map(lambda x: preprocess(x), sentences))
    print("Pre-processing done")

    # get average word vectors from glove
    print("Extracting GloVe word vector feature")
    glove_features = get_glove(pre_processed_sentences)
    print("Finished GloVe feature extraction")
    features['glove'] = glove_features

    # get tf-idf vectors
    print("Extracting tf-idf vector feature")
    tf_idf_features = get_tf_idf(pre_processed_sentences, pre_processed_corpus)
    print("Finished tf-idf feature extraction")
    features['tf_idf'] = tf_idf_features

    # get word n-gram vectors
    print("Extracting word n-grams vector feature")
    word_ngram_features = get_word_ngrams(pre_processed_sentences, pre_processed_corpus)
    print("Finished word n-grams feature extraction")
    features['n_gram'] = word_ngram_features

    # get POS counts
    print("Extracting POS counts features")
    pos_counts_features = get_pos_counts(sentences)
    print("Finished POS counts feature extraction")
    features.update(pos_counts_features)

    return features

if __name__ == '__main__':
    corpus, _, _ = read_data_from_dir('./data/dailydialog', split="train")
    utterances, emotions, _ = read_data_from_dir('./data/dailydialog', split="test")
    features = extract_features(utterances, corpus)

    with open('./data/dailydialog/features.pkl', 'wb') as f:
        pickle.dump(features, f)
