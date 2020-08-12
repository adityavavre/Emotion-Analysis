import argparse
import logging
import os
import time
from typing import List
import pickle
import json

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

from utils import read_data_from_dir


def train_model(model, X_train, Y_train, X_valid, Y_valid, X_test, Y_test, params, labels):
    print("Setting model params: ", params)
    model = model.set_params(**params)
    print("Starting fit")
    start_time = time.time()

    model = model.fit(X_train, Y_train)

    duration = time.time() - start_time
    print("Fit done")
    print("Total training time: %s seconds", duration)

    Y_pred_proba = model.predict_proba(X_train)
    Y_pred = np.argmax(Y_pred_proba, axis=1)
    train_report = classification_report(Y_train, Y_pred, target_names=labels)
    train_acc = accuracy_score(Y_train, Y_pred)
    print("Train Accuracy: ", train_acc)
    print(train_report)

    Y_pred_proba = model.predict_proba(X_valid)
    Y_pred = np.argmax(Y_pred_proba, axis=1)
    valid_report = classification_report(Y_valid, Y_pred, target_names=labels)
    valid_acc = accuracy_score(Y_valid, Y_pred)
    print("Validation Accuracy: ", valid_acc)
    print(valid_report)

    Y_pred_proba = model.predict_proba(X_test)
    Y_pred = np.argmax(Y_pred_proba, axis=1)
    test_report = classification_report(Y_test, Y_pred, target_names=labels)
    test_acc = accuracy_score(Y_test, Y_pred)
    print("Test Accuracy: ", test_acc)
    print(test_report)

    final_report = {
        "train_accuracy": train_acc,
        "train_report": train_report,
        "validation_accuracy": valid_acc,
        "validation_report": valid_report,
        "test_accuracy": test_acc,
        "test_report": test_report
    }

    return model, final_report

def run_models(models_list: List,
               models_names: List[str],
               X_train, Y_train,
               X_valid, Y_valid,
               X_test, Y_test,
               params_list: List,
               labels: List,
               out_dir: str):

    for name, model, params in zip(models_names, models_list, params_list):
        print("Executing model: ", name)
        trained_model, report = train_model(model, X_train, Y_train, X_valid, Y_valid, X_test, Y_test, params, labels)
        model_path = os.path.join(out_dir, name, 'model.pkl')
        report_file = os.path.join(out_dir, name, 'report.json')

        print("Saving trained model to: ", model_path)
        with open(model_path, 'wb') as f:
            pickle.dump(trained_model, f)

        print("Saving training report to: ", report_file)
        with open(report_file, 'w') as f:
            json.dump(report, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Argument parser for regressors")
    # parser.add_argument('-m', '--model', action="store", dest="model", type=str, required=True)
    parser.add_argument('-d', '--data', action="store", dest="data", type=str, required=True)
    parser.add_argument('-o', '--out_dir', action="store", dest="out_dir", type=str, required=True)
    parser.add_argument('-t', '--train', default=False, dest="train", action='store_true')

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    args = parser.parse_args()
    # model = args.model
    data_dir = args.data
    do_train = args.train
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # labels specified in the dataset in the given order
    labels = ["no emotion", "anger", "disgust", "fear", "happiness", "sadness", "surprise"]

    train_file = os.path.join(data_dir, 'train_features.csv')
    valid_file = os.path.join(data_dir, 'validation_features.csv')
    test_file = os.path.join(data_dir, 'test_features.csv')

    X_train = pd.read_csv(train_file, skiprows=0).to_numpy()
    X_valid = pd.read_csv(valid_file, skiprows=0).to_numpy()
    X_test = pd.read_csv(test_file, skiprows=0).to_numpy()

    _, Y_train, _ = read_data_from_dir('./data/dailydialog', split="train")
    _, Y_valid, _ = read_data_from_dir('./data/dailydialog', split="validation")
    _, Y_test, _ = read_data_from_dir('./data/dailydialog', split="test")

    Y_train, Y_valid, Y_test = np.array(Y_train), np.array(Y_valid), np.array(Y_test)

    models = [LogisticRegression()]
    models_names = ['logistic_regression']
    params_list = [{'verbose': False, 'max_iter': 1e8}]

    run_models(models, models_names, X_train, Y_train, X_valid, Y_valid, X_test, Y_test, params_list=params_list, labels=labels, out_dir=out_dir)
