import csv
import os
import numpy as np
from nltk import word_tokenize, PorterStemmer
from nltk.corpus import stopwords
from num2words import num2words
import pandas as pd
import nltk.data
from ast import literal_eval

def read_data_from_dir(in_dir: str, split: str):
    """

    :param in_dir:
    :param split:
    :return:
    """
    if split not in ("train", "test", "validation"):
        raise Exception("Not a valid split. Must be one of `train`, `test` or `validation`")

    dial_dir = os.path.join(in_dir, split, 'dialogues_'+split+'.txt')
    emo_dir = os.path.join(in_dir, split, 'dialogues_emotion_'+split+'.txt')
    act_dir = os.path.join(in_dir, split, 'dialogues_act_'+split+'.txt')

    # Open files
    in_dial = open(dial_dir, 'r', encoding='utf-8')
    in_emo = open(emo_dir, 'r', encoding='utf-8')
    in_act = open(act_dir, 'r', encoding='utf-8')

    utterances = []
    act_annotations = []
    emotions = []

    for line_count, (line_dial, line_emo, line_act) in enumerate(zip(in_dial, in_emo, in_act)):
        seqs = line_dial.strip().split('__eou__')
        seqs = seqs[:-1]
        seqs = list(map(lambda x: x[1:] if x[0]==' ' else x, seqs))
        seqs = list(map(lambda x: x[:-1] if x[-1] == ' ' else x, seqs))

        emos = line_emo.strip().split(' ')
        emos = list(map(lambda x: int(x), emos))

        acts = line_act.strip().split(' ')
        acts = list(map(lambda x: int(x), acts))

        seq_len = len(seqs)
        emo_len = len(emos)
        act_len = len(acts)
        if seq_len != emo_len or seq_len != act_len:
            print("Different turns btw dialogue & emotion & action! ",
                  line_count + 1, seq_len, emo_len, act_len)
            continue

        utterances.extend(seqs)
        act_annotations.extend(acts)
        emotions.extend(emos)

    assert len(utterances) == len(act_annotations) and len(utterances) == len(emotions), \
        "Length of utterances do not match acts or emotions"
    print("Number of examples in "+split+ " set: ", len(utterances))

    return utterances, emotions, act_annotations

def read_meld_data(in_dir: str, split: str):
    """

    :param in_dir:
    :param split:
    :return:
    """
    if split not in ("train", "test", "dev"):
        raise Exception("Not a valid split. Must be one of `train`, `test` or `dev`")

    labels = ["neutral", "anger", "disgust", "fear", "joy", "sadness", "surprise"]
    label2id = {v: i for i, v in enumerate(labels)}

    data_file = os.path.join(in_dir, split+"_sent_emo.csv")
    df = pd.read_csv(data_file, encoding='utf-8', error_bad_lines=False)

    utterances = list(df['Utterance'])
    utterances = list(map(lambda x: x.encode('ascii', 'ignore').decode('utf-8'), utterances))
    emotions = list(df['Emotion'])
    emotions = list(map(lambda x: label2id[x], emotions))

    assert len(utterances) == len(emotions), "Length of utterances do not match length of emotions"
    print("Number of examples in " + split + " set: ", len(utterances))

    return utterances, emotions

def convert_lower_case(data):
    return np.char.lower(data)

def remove_stop_words(data):
    stop_words = stopwords.words('english')
    words = word_tokenize(str(data))
    new_text = ""
    for w in words:
        if w not in stop_words and len(w) > 1:
            new_text = new_text + " " + w
    return new_text

def convert_numbers(data):
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        try:
            w = num2words(int(w))
        except:
            a = 0
        new_text = new_text + " " + w
    new_text = np.char.replace(new_text, "-", " ")
    return new_text

def remove_punctuation(data):
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    for i in range(len(symbols)):
        data = np.char.replace(data, symbols[i], ' ')
        data = np.char.replace(data, "  ", " ")
    data = np.char.replace(data, ',', '')
    return data

def remove_apostrophe(data):
    return np.char.replace(data, "'", "")


def stemming(data):
    stemmer = PorterStemmer()
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        new_text = new_text + " " + stemmer.stem(w)
    return new_text

def preprocess(data):
    data = convert_lower_case(data)
    data = remove_punctuation(data)
    data = remove_apostrophe(data)
    data = convert_numbers(data)
    data = remove_punctuation(data)
    # data = convert_numbers(data)
    # data = remove_punctuation(data)
    return str(data)

class Tokenizer():
    def __init__(self):
        self.tokenizer = word_tokenize
    def tokenize(self, x):
        return self.tokenizer(x)

def read_reddit_data(filename: str):
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            post = line.strip()
            if len(post) < 20:
                continue
            temp_sentences = sent_detector.tokenize(post)
            for s in temp_sentences:
                if len(s.split()) < 3:
                    continue
                sentences.append(preprocess(s))
    return list(set(sentences))

if __name__ == '__main__':
    # a, b, c = read_data_from_dir('./data/dailydialog/', 'train')
    # print(a[:2])
    # print(b[:2])
    # print(c[:2])

    # ut, em = read_meld_data('./data/meld', 'train')
    # print(ut[:2])
    # print(em[:2])

    sents = read_reddit_data('./reddit_scraper/reddit.csv')
    print("Len: ", len(sents))
    print("sents: ", sents[:10])
