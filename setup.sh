#!/usr/bin/env bash

mkdir distilbert data

wget https://cdn.huggingface.co/distilbert-base-uncased-pytorch_model.bin -O ./distilbert/pytorch_model.bin
wget https://s3.amazonaws.com/models.huggingface.co/bert/distilbert-base-uncased-config.json -O ./distilbert/config.json
wget https://cdn.huggingface.co/distilbert-base-uncased-modelcard.json -O ./distilbert/modelcard.json

wget http://yanran.li/files/ijcnlp_dailydialog.zip -O ./data/ijcnlp_dailydialog.zip
unzip ./data/ijcnlp_dailydialog.zip -d ./data/
mv ./data/ijcnlp_dailydialog ./data/dailydialog
unzip ./data/dailydialog/train.zip -d ./data/dailydialog/
unzip ./data/dailydialog/validation.zip -d ./data/dailydialog/
unzip ./data/dailydialog/test.zip -d ./data/dailydialog/

mkdir -p ./data/meld/
wget https://github.com/declare-lab/MELD/raw/master/data/MELD/dev_sent_emo.csv -O ./data/meld/dev_sent_emo.csv
wget https://github.com/declare-lab/MELD/raw/master/data/MELD/test_sent_emo.csv -O ./data/meld/test_sent_emo.csv
wget https://github.com/declare-lab/MELD/raw/master/data/MELD/train_sent_emo.csv -O ./data/meld/train_sent_emo.csv
