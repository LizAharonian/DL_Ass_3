#!/bin/bash

echo run model: d on: pos
python bilstmTrain.py d pos/train modelFile_pos_d pos pos/dev

echo run model: a on: ner
python bilstmTrain.py a ner/train modelFile_ner_a ner ner/dev
echo run model: b on: ner
python bilstmTrain.py b ner/train modelFile_ner_b ner ner/dev
echo run model: c on: ner
python bilstmTrain.py c ner/train modelFile_ner_c ner ner/dev
echo run model: d on: ner
python bilstmTrain.py d ner/train modelFile_ner_d ner ner/dev