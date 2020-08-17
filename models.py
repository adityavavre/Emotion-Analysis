import argparse
import logging
import os
import time
from typing import List
import pickle
import json

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, log_loss
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import ParameterGrid

from utils import read_data_from_dir


def grid_search(model, X_train, Y_train, X_valid, Y_valid, params):
    print("Executing GridSearch over params: ", params)
    best_params = None
    best_loss = np.inf
    for param in ParameterGrid(params):
        print("Fitting with params: ", param)
        model = model.set_params(**param)
        model = model.fit(X_train, Y_train)
        Y_pred = model.predict_proba(X_valid)
        loss = log_loss(Y_valid, Y_pred)
        print("Loss: ", loss)
        if loss < best_loss:
            print("Found better param")
            best_params = param
            best_loss = loss
    print("Completed grid search")

    ## fit the model again with best params and return
    print("Fitting model with best params found")
    model = model.set_params(**best_params)
    model = model.fit(X_train, Y_train)
    print("Fit done")
    return model, best_params


def get_report(model, X, Y_true, labels: List, split: str):
    Y_pred_proba = model.predict_proba(X)
    Y_pred = np.argmax(Y_pred_proba, axis=1)
    report = classification_report(Y_true, Y_pred, target_names=labels, output_dict=True)
    acc = accuracy_score(Y_true, Y_pred)
    print(split+" accuracy: ", acc)
    print(classification_report(Y_true, Y_pred, target_names=labels))
    return report, acc

def train_model(model, X_train, Y_train, X_valid, Y_valid, X_test, Y_test, params, labels):
    # print("Setting model params: ", params)
    # model = model.set_params(**params)
    # print("Starting fit")
    start_time = time.time()

    model, best_params = grid_search(model, X_train, Y_train, X_valid, Y_valid, params)

    duration = time.time() - start_time
    # print("Fit done")
    print("Total time taken: %s seconds" % duration)

    train_report, train_acc = get_report(model, X_train, Y_train, labels=labels, split="Train")

    valid_report, valid_acc = get_report(model, X_valid, Y_valid, labels=labels, split="Validation")

    test_report, test_acc = get_report(model, X_test, Y_test, labels=labels, split="Test")

    final_report = {
        "train_accuracy": train_acc,
        "train_report": train_report,
        "validation_accuracy": valid_acc,
        "validation_report": valid_report,
        "test_accuracy": test_acc,
        "test_report": test_report,
        "best_params": best_params
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
    assert len(models_list) == len(models_names) and len(models_list) == len(params_list), \
        "The models, names and params list must have the same length"

    for name, model, params in zip(models_names, models_list, params_list):
        print("Executing model: ", name)
        trained_model, report = train_model(model, X_train, Y_train, X_valid, Y_valid, X_test, Y_test, params, labels)
        os.makedirs(os.path.join(out_dir, name), exist_ok=True)
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

    X_train = pd.read_csv(train_file, skiprows=0, usecols=lambda col: col not in ["emotions"]).to_numpy()
    X_valid = pd.read_csv(valid_file, skiprows=0, usecols=lambda col: col not in ["emotions"]).to_numpy()
    X_test = pd.read_csv(test_file, skiprows=0, usecols=lambda col: col not in ["emotions"]).to_numpy()

    Y_train = pd.read_csv(train_file, skiprows=0, usecols=["emotions"]).to_numpy()
    Y_valid = pd.read_csv(valid_file, skiprows=0, usecols=["emotions"]).to_numpy()
    Y_test = pd.read_csv(test_file, skiprows=0, usecols=["emotions"]).to_numpy()

    print(X_test.shape)
    print(X_test)

    print(Y_test.shape)
    print(Y_test)
    exit()
    # _, Y_train, _ = read_data_from_dir('./data/dailydialog', split="train")
    # _, Y_valid, _ = read_data_from_dir('./data/dailydialog', split="validation")
    # _, Y_test, _ = read_data_from_dir('./data/dailydialog', split="test")

    Y_train, Y_valid, Y_test = np.array(Y_train), np.array(Y_valid), np.array(Y_test)

    models = [
        # LogisticRegression(),
        # DecisionTreeClassifier(),
        # RandomForestClassifier(),
        # XGBClassifier(),
        AdaBoostClassifier()
    ]
    models_names = [
        # 'logistic_regression',
        # "decision_tree",
        # "random_forest",
        # "xgb",
        "adaboost"
    ]
    # params_list = []
    params_list = [
        # {
        #     'verbose': False, 'max_iter': 1e8
        # },
        # {
        #     "max_depth": [ 5, 8],
        #     "min_samples_leaf": [10, 15]
        # },
        # {
        #     "n_estimators": [200, 400],
        #     "max_depth": [4, 7],
        #     "min_samples_leaf": [5, 7]
        # },
        # {
        #     "learning_rate": [0.1, 0.01],
        #     "n_estimators": [200, 400],
        #     "max_depth": [4, 7]
        # },
        {
            "base_estimator": [DecisionTreeClassifier(max_depth=8,
                                                    min_samples_leaf=6),
                               DecisionTreeClassifier(max_depth=10,
                                                    min_samples_leaf=8)],
            "n_estimators": [50, 100],
            "learning_rate": [0.01, 0.1]
        }
    ]

    run_models(models,
               models_names,
               X_train, Y_train,
               X_valid, Y_valid,
               X_test, Y_test,
               params_list=params_list,
               labels=labels,
               out_dir=out_dir)
