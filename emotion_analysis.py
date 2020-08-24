import argparse
import os
from collections import defaultdict
from typing import List, Dict, Tuple
from utils import read_data_from_dir, read_meld_data, read_reddit_data
import logging

import numpy as np
import pandas as pd
import pkbar
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer, BertModel, \
    get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)


class DailyDialogDataset(Dataset):
    def __init__(self, utterances: List, targets: List, tokenizer: PreTrainedTokenizer, max_len: int = 128):
        self.utterances = utterances
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, item):
        utterance = str(self.utterances[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
            utterance,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'utterance': utterance,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }


def create_data_loader(data_dir: str,
                       split: str,
                       tokenizer: PreTrainedTokenizer,
                       max_len: int,
                       batch_size: int,
                       combined: bool = False) -> Tuple[DataLoader, int]:
    """

    :param data_dir:
    :param split:
    :param tokenizer:
    :param max_len:
    :param batch_size:
    :return:
    """
    utterances, emotions, _ = read_data_from_dir(os.path.join(data_dir, 'dailydialog'), split=split)
    if combined:
        meld_utterances, meld_emotions = read_meld_data(
            os.path.join(data_dir, 'meld'),
            split="dev" if split=="validation" else split)

        utterances.extend(meld_utterances)
        emotions.extend(meld_emotions)

    ds = DailyDialogDataset(
        utterances=utterances,
        targets=emotions,
        tokenizer=tokenizer,
        max_len=max_len)

    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4), len(utterances)


class EmotionClassifier(nn.Module):
    """

    """

    def __init__(self, model_name_or_path: str, n_classes: int):
        """

        :param model_name_or_path:
        :param n_classes:
        """
        super(EmotionClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name_or_path)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        """

        :param input_ids:
        :param attention_mask:
        :return:
        """
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        output = self.drop(pooled_output)
        return self.out(output)


class EmotionAnalyser():
    """

    """

    def __init__(self, model, device):
        """

        :param model:
        """
        self.device = device
        self.model = model.to(self.device)

    def train_epoch(
            self,
            data_loader: DataLoader,
            loss_fn,
            optimizer,
            scheduler,
            n_examples: int,
            kbar: pkbar.Kbar):
        """

        :param data_loader:
        :param loss_fn:
        :param optimizer:
        :param scheduler:
        :param n_examples:
        :param kbar:
        :return:
        """
        model = self.model.train()
        losses = []
        correct_predictions = 0
        i = 0
        for d in data_loader:
            input_ids = d["input_ids"].to(self.device)
            attention_mask = d["attention_mask"].to(self.device)
            targets = d["targets"].to(self.device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            kbar.update(i, values=[("loss", loss.item())])
            i += 1

        return correct_predictions.double() / n_examples, np.mean(losses)

    def eval_model(self, data_loader: DataLoader, loss_fn, n_examples: int):
        """

        :param data_loader:
        :param loss_fn:
        :param n_examples:
        :return:
        """
        model = self.model.eval()

        losses = []
        correct_predictions = 0

        with torch.no_grad():
            for d in data_loader:
                input_ids = d["input_ids"].to(self.device)
                attention_mask = d["attention_mask"].to(self.device)
                targets = d["targets"].to(self.device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                _, preds = torch.max(outputs, dim=1)

                loss = loss_fn(outputs, targets)

                correct_predictions += torch.sum(preds == targets)
                losses.append(loss.item())

        return correct_predictions.double() / n_examples, np.mean(losses)

    def train(
            self,
            train_data_loader: DataLoader,
            train_num_examples: int,
            val_data_loader: DataLoader,
            val_num_examples: int,
            output_dir: str = './saved_models/',
            num_epochs: int = 2,
            lr: float = 2e-5):
        """

        :param train_data_loader:
        :param val_data_loader:
        :param num_epochs:
        :param lr:
        :return:
        """

        history = defaultdict(list)
        best_accuracy = 0
        optimizer = AdamW(model.parameters(), lr=lr)
        total_steps = len(train_data_loader) * num_epochs

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        loss_fn = nn.CrossEntropyLoss().to(self.device)

        print("Running " + str(num_epochs) + " training epochs\n")
        for epoch in range(num_epochs):
            print('Epoch: %d/%d' % (epoch + 1, num_epochs))
            kbar = pkbar.Kbar(target=len(train_data_loader), width=35)

            train_acc, train_loss = self.train_epoch(
                train_data_loader,
                loss_fn,
                optimizer,
                scheduler,
                train_num_examples,
                kbar
            )

            val_acc, val_loss = self.eval_model(
                val_data_loader,
                loss_fn,
                val_num_examples
            )

            history['train_acc'].append(train_acc)
            history['train_loss'].append(train_loss)
            history['val_acc'].append(val_acc)
            history['val_loss'].append(val_loss)

            if val_acc > best_accuracy:
                logger.info("")
                os.makedirs(output_dir, exist_ok=True)
                torch.save(self.model, os.path.join(output_dir, 'pytorch_model.bin'))
                best_accuracy = val_acc

            kbar.add(1, values=[("loss", train_loss), ("acc. ", train_acc),
                                ("val_loss", val_loss), ("val_acc. ", val_acc)])

    def get_predictions(self, data_loader: DataLoader, id2label: Dict):
        """

        :param data_loader:
        :param id2label:
        :return:
        """
        model = self.model.eval()

        utterances = []
        predictions = []
        prediction_probs = []
        real_values = []

        with torch.no_grad():
            for d in data_loader:
                texts = d["utterance"]
                input_ids = d["input_ids"].to(self.device)
                attention_mask = d["attention_mask"].to(self.device)
                targets = d["targets"].to(self.device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                _, preds = torch.max(outputs, dim=1)

                probs = F.softmax(outputs, dim=1)

                utterances.extend(texts)
                predictions.extend(preds)
                prediction_probs.extend(probs)
                real_values.extend(targets)

        predictions = torch.stack(predictions).cpu()
        prediction_probs = torch.stack(prediction_probs).cpu()
        real_values = torch.stack(real_values).cpu()
        y_test = real_values
        y_pred = predictions
        labels = [id2label[i] for i in range(len(id2label))]
        print("Test results: ")
        print(classification_report(y_test, y_pred, target_names=labels))
        print("id2label: \n", id2label)
        print("Confusion matrix: ")
        print(confusion_matrix(y_test, y_pred, labels=[i for i in range(len(id2label))]))
        return utterances, predictions, prediction_probs, real_values

    # def predict(
    #         self,
    #         summary: str,
    #         tokenizer: PreTrainedTokenizer,
    #         max_len: int,
    #         id2label: Dict):
    #     """
    #
    #     :param summary:
    #     :param tokenizer:
    #     :param max_len:
    #     :param id2label:
    #     :return:
    #     """
    #     encoded_summary = tokenizer.encode_plus(
    #         summary,
    #         max_length=max_len,
    #         add_special_tokens=True,
    #         return_token_type_ids=False,
    #         pad_to_max_length=True,
    #         return_attention_mask=True,
    #         return_tensors='pt',
    #     )
    #     input_ids = encoded_summary['input_ids'].to(self.device)
    #     attention_mask = encoded_summary['attention_mask'].to(self.device)
    #
    #     output = self.model(input_ids, attention_mask)
    #     _, prediction = torch.max(output, dim=1)
    #
    #     print(f'Summary text: {summary}')
    #     print(f'Sentiment  : {id2label[prediction.item()]}')
    #     print()
    #     return id2label[prediction.item()]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Argument parser for pipeline")
    parser.add_argument('-m', '--model', action="store", dest="model", type=str, required=True)
    parser.add_argument('-d', '--data', action="store", dest="data", type=str, required=True)
    parser.add_argument('-t', '--train', default=False, dest="train", action='store_true')
    parser.add_argument('-ts', '--test', default=False, dest="test", action='store_true')
    parser.add_argument('-p', '--pred', default=False, dest="pred", action='store_true')

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    args = parser.parse_args()
    model_dir = args.model
    data_dir = args.data
    do_train = args.train
    do_test = args.test
    do_pred = args.pred

    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    labels = ["no emotion", "anger", "disgust", "fear", "happiness", "sadness", "surprise"]
    label2id = {v: i for i, v in enumerate(labels)}
    id2label = {i: v for i, v in enumerate(labels)}


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    if do_train:
        model = EmotionClassifier(model_dir, len(labels))
        analyser = EmotionAnalyser(model, device)

        MAX_LEN = 64
        BATCH_SIZE = 32
        EPOCHS = 10
        LEARNING_RATE = 2e-5

        train_data_loader, train_num_examples = create_data_loader(data_dir,
                                                                   "train",
                                                                   tokenizer,
                                                                   max_len=MAX_LEN,
                                                                   batch_size=BATCH_SIZE,
                                                                   combined=True)

        val_data_loader, val_num_examples = create_data_loader(data_dir,
                                                               "validation",
                                                               tokenizer,
                                                               max_len=MAX_LEN,
                                                               batch_size=BATCH_SIZE,
                                                               combined=True)

        test_data_loader, test_num_examples = create_data_loader(data_dir,
                                                                 "test",
                                                                 tokenizer,
                                                                 max_len=MAX_LEN,
                                                                 batch_size=BATCH_SIZE,
                                                                 combined=True)

        analyser.train(train_data_loader,
                       train_num_examples,
                       val_data_loader,
                       val_num_examples,
                       num_epochs=EPOCHS,
                       lr=LEARNING_RATE)

        ## predictions on test set
        utterances, predictions, prediction_probs, real_values = analyser.get_predictions(test_data_loader, id2label)
        print("utterances: ", len(utterances))
        print("predictions: ", predictions)
        print("real_values: ", real_values)

        os.makedirs('./predictions/', exist_ok=True)
        df_out = pd.DataFrame(
            list(zip(
                utterances,
                list(map(lambda x: id2label[x], predictions.tolist())),
                list(map(lambda x: id2label[x], real_values.tolist()))
            )),
            columns=['Utterance', 'Predicted Emotion', 'Actual Emotion'])

        df_out.to_csv('./predictions/test_pred.csv', index=False)

    elif do_test:
        MAX_LEN = 64
        BATCH_SIZE = 32
        model  = torch.load(model_dir)
        # model.load_state_dict(torch.load('./finetuned_bert/best_model_state.bin'))
        analyzer = EmotionAnalyser(model, device)

        test_data_loader, test_num_examples = create_data_loader(data_dir,
                                                                 "test",
                                                                 tokenizer,
                                                                 max_len=MAX_LEN,
                                                                 batch_size=BATCH_SIZE,
                                                                 combined=False)

        utterances, predictions, prediction_probs, real_values = analyzer.get_predictions(test_data_loader, id2label)

    elif do_pred:
        MAX_LEN = 64
        BATCH_SIZE = 32

        model = torch.load(model_dir)
        analyzer = EmotionAnalyser(model, device)

        sentences = read_reddit_data(data_dir)
        ds = DailyDialogDataset(
            utterances=sentences,
            targets=[0]*len(sentences),
            tokenizer=tokenizer,
            max_len=MAX_LEN)
        dl =  DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            num_workers=4)

        utterances, predictions, _, _ = analyzer.get_predictions(dl, id2label)
        os.makedirs('./predictions/', exist_ok=True)
        df_out = pd.DataFrame(
            list(zip(
                utterances,
                list(map(lambda x: id2label[x], predictions.tolist()))
            )),
            columns=['Utterance', 'Predicted Emotion'])
        df_out.to_csv('./predictions/reddit_pred.csv', index=False)

    else:
        print("No command specified!\n Exiting!")
    exit(0)
