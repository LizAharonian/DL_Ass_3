import sys
import os
from time import time
import random as rand
import numpy as np
import dynet as dy
import utils_part_3 as ut
import BILSTMNeuralNets as nn
import cPickle as pickle
from zipfile import ZipFile

STUDENT = {'name': "Liz Aharonian_Ori Ben Zaken",
           'ID': "316584960_311492110"}

EPOCHS = 5
TRIANED_EXAMPLES_UNTIL_DEV = 500
MODEL = "POS"


def compute_accuracy(model, tagged_data, type):
    """
    Computes the accuracy of the model on the tagged data.
    :param model: bi-lstm model
    :param tagged_data: list of examples, each example (sentence) is a list of (word,tag)
    :param type: pos/ner
    :return: accuracy
    """
    good = 0
    total_words = 0
    for tagged_sentence in tagged_data:
        words, tags = split_sentence_to_words_and_tags(tagged_sentence)
        preds = model.get_prediction_on_sentence(words)
        for pred, tag in zip(preds, tags):
            # we don't consider correct taggings of Other ("O") label on
            # ner data as good predictions
            if type == "ner" and pred == "O" and tag == "O":
                pass
            elif pred == tag:
                good += 1
        total_words += len(words)
    return float(good) / float(total_words) * 100

def split_sentence_to_words_and_tags(tagged_sentence):
    """
    Split tagged_sentence which is a list of (word,tag) to two lists of: words, tags
    :param tagged_sentence: tagged sentence example
    :return: words, tags lists
    """
    words = [word for word, tag in tagged_sentence]
    tags = [tag for word, tag in tagged_sentence]
    return words, tags

def train(model, train_data, dev_data, type, rep):
    """
    Trains the model over the train_data for EPOCHS epochs.
    Every TRIANED_EXAMPLES_UNTIL_DEV, we let the model go over the dev_data
    :param model: bi-lstm model
    :param train_data: train data
    :param dev_data: dev data
    :param type: pos/ner
    :param rep: a/b/c/d - model type
    """
    trainer = model.trainer
    graph ={}
    # training
    start_time = time()
    for epoch in range(EPOCHS):
        losses_list = []
        print "Epoch number " + str(epoch) + " started!"
        # training iter
        rand.shuffle(train_data)
        i = 1
        for tagged_sentence in train_data:
            words, tags = split_sentence_to_words_and_tags(tagged_sentence)
            loss = model.get_train_loss(words, tags)
            losses_list.append(loss.value())
            loss.backward()
            trainer.update()

            if i % TRIANED_EXAMPLES_UNTIL_DEV == 0:
                # get accuracy on the dev data
                acc = str(compute_accuracy(model, dev_data, type))
                print "dev results: " + " accuracy is: " +  acc + "%"
                # save the accurcy result to the graph data
                graph[i/100] = acc
            i += 1
        # compute average loss
        avg_loss = np.average(losses_list)
        # end of one iter on train data
        print "train epoch" + str(epoch) + "results: " + "loss is: " + str(float(avg_loss) / len(train_data)) + \
              " accuracy is: " + str(compute_accuracy(model, train_data, type)) + "%"
    end_time = time()
    total_time = end_time - start_time
    print "total time: " + str(total_time)
    # save the graph data to binary file
    with open(rep + "_model_" + type + ".pkl", "wb") as output:
        pickle.dump(graph, output, pickle.HIGHEST_PROTOCOL)

def save_model(model, model_file):
    """
    Saves the model and its dependencies to model_file
    :param model: bi-lstm model
    :param model_file: name of the file for saving the model
    """
    # save the indexers of words, tags, chars, prefixes and suffixes sets in dicts.pkl
    with open("dicts.pkl", "wb") as output:
        pickle.dump(ut.W2I, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(ut.T2I, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(ut.C2I, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(ut.W2I, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(ut.I2T, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(ut.I2C, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(ut.P2I, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(ut.S2I, output, pickle.HIGHEST_PROTOCOL)
    # save the dy-net model in model.dy
    model.model.save("model.dy")
    zip_file = ZipFile(model_file, "w")
    # zip dicts.pkl and model.dy
    zip_file.write("dicts.pkl")
    zip_file.write("model.dy")
    zip_file.close()
    # remove the files after writing them to the model_file zip
    os.remove("dicts.pkl")
    os.remove("model.dy")


def main(repr, train_file, model_file, type, dev_file=None):
    train_data = ut.read_tagged_data(train_file)
    ut.load_indexers()
    if dev_file:
        dev_data = ut.read_tagged_data(dev_file, is_dev=True)
    # take 20% of the train data for dev
    else:
        eighty_prec_len = int(len(train_data) * 0.8)
        train_data, dev_data = train_data[:eighty_prec_len], train_data[eighty_prec_len:]

    # Initialize model
    if repr == "a":
        model = nn.Model_A(repr,ut.T2I, ut.W2I, ut.I2T)
    elif repr == "b":
        model = nn.Model_B(repr,ut.T2I, ut.W2I, ut.I2T, ut.C2I)
    elif repr == "c":
        model = nn.Model_C(repr,ut.T2I, ut.W2I, ut.I2T, ut.P2I, ut.S2I)
    elif repr == "d":
        model = nn.Model_D(repr,ut.T2I, ut.W2I, ut.I2T, ut.C2I)
    else:
        print("Unvalid repr. Program quits")
        sys.exit(1)
    train(model, train_data, dev_data, type, repr)
    save_model(model, model_file)

if __name__ == "__main__":
    main(*sys.argv[1:])