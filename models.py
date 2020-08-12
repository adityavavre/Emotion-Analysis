import argparse
import logging
import os
import pandas as pd
import numpy as np

from utils import read_data_from_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Argument parser for regressors")
    parser.add_argument('-m', '--model', action="store", dest="model", type=str, required=True)
    parser.add_argument('-d', '--data', action="store", dest="data", type=str, required=True)
    parser.add_argument('-t', '--train', default=False, dest="train", action='store_true')

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    args = parser.parse_args()
    model = args.model
    data_dir = args.data
    do_train = args.train

    # labels specified in the dataset in the given order
    labels = ["no emotion", "anger", "disgust", "fear", "happiness", "sadness", "surprise"]

    train_file = os.path.join(data_dir, 'train_features.csv')
    valid_file = os.path.join(data_dir, 'validation_features.csv')
    test_file = os.path.join(data_dir, 'test_features.csv')

    X_train = pd.read_csv(train_file, skiprows=1).to_numpy()
    X_valid = pd.read_csv(valid_file, skiprows=1).to_numpy()
    X_test = pd.read_csv(test_file, skiprows=1).to_numpy()

    _, Y_train, _ = read_data_from_dir('./data/dailydialog', split="train")
    _, Y_valid, _ = read_data_from_dir('./data/dailydialog', split="validation")
    _, Y_test, _ = read_data_from_dir('./data/dailydialog', split="test")

    Y_train, Y_valid, Y_test = np.array(Y_train), np.array(Y_valid), np.array(Y_test)

