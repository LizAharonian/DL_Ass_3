import sys
from time import time
import random as rand
import dynet as dy
import utils_part_3 as ut
import BILSTMNeuralNets as nn
import cPickle as pickle
from zipfile import ZipFile

EPOCHS = 5
TRIANED_EXAMPLES_UNTIL_DEV = 500
MODEL = "POS"


def compute_accuracy(model, tagged_data, type):
    good = 0
    total_words = 0
    for tagged_sentence in tagged_data:
        words, tags = split_sentence_to_words_and_tags(tagged_sentence)
        preds = model.get_prediction_on_sentence(words)
        #preds = [ut.I2T[pred] for pred in preds]
        for pred, tag in zip(preds, tags):
            if type == "ner" and pred == "O" and tag == "O":
                pass
            elif pred == tag:
                good += 1
        total_words += len(words)
    return float(good) / float(total_words) * 100

def split_sentence_to_words_and_tags(tagged_sentence):
    words = [word for word, tag in tagged_sentence]
    tags = [tag for word, tag in tagged_sentence]
    return words, tags

# def dev(model, dev_data):
#     sum_of_losses = 0.0
#     for tagged_sentence in dev_data:
#         words, tags = split_sentence_to_words_and_tags(words, tags)
#         preds = model.get_prediction_on_sentence(words)
#         preds = [ut.I2T[pred] for pred in preds]
#
#         sum_of_losses += loss.npvalue()
#         loss.backward()
#         # end of one iter on test data
#     print "dev results: " + "loss is: " + str(float(sum_of_losses) / len(dev_data)) + " accuracy is: " + \
#           str(compute_accuracy(dev_data))

def train(model, train_data, dev_data, type):
    trainer = dy.AdamTrainer(model.model)
    # training
    sum_of_losses = 0.0
    start_time = time()
    for epoch in range(EPOCHS):
        print "Epoch number " + str(epoch) + " started!"
        # training iter
        rand.shuffle(train_data)
        i = 1
        for tagged_sentence in train_data:
            words, tags = split_sentence_to_words_and_tags(tagged_sentence)
            loss = model.get_train_loss(words, tags)
            sum_of_losses += loss.npvalue()
            loss.backward()
            trainer.update()

            if i % TRIANED_EXAMPLES_UNTIL_DEV == 0:
                print "dev results: " + " accuracy is: " + str(compute_accuracy(model, dev_data, type)) + "%"
            i += 1

        # end of one iter on train data
        print "train epoch" + epoch + "results: " + "loss is: " + str(float(sum_of_losses) / len(train_data)) + \
              " accuracy is: " + str(compute_accuracy(model, train_data, type) + "%")
        sum_of_losses = 0.0
    end_time = time()
    total_time = end_time - start_time
    print "total time: " + str(total_time)

def save_model(model, model_file):
    with open("dicts.pkl", "wb") as output:
        pickle.dump(ut.W2I, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(ut.T2I, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(ut.C2I, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(ut.W2I, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(ut.I2T, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(ut.I2C, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(ut.PREFIXES, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(ut.SUFFIXES, output, pickle.HIGHEST_PROTOCOL)

    model.model.save("model.dy")
    zip_file = ZipFile(model_file, "w")
    zip_file.write("dicts.pkl")
    zip_file.write("model.dy")
    zip_file.close()


def main(repr, train_file, model_file, type, dev_file=None):
    train_data = ut.read_tagged_data(train_file)
    ut.load_indexers()
    if dev_file:
        dev_data = ut.read_tagged_data(dev_file, is_dev=True)
    else:
        eighty_prec_len = int(len(train_data) * 0.8)
        train_data, dev_data = train_data[:eighty_prec_len], train_data[eighty_prec_len:]

    # Initialize model
    if repr == "a":
        model = nn.Model_A(ut.T2I, ut.W2I, ut.I2T)
    elif repr == "b":
        model = nn.Model_B(ut.T2I, ut.W2I, ut.I2T, ut.C2I)
    elif repr == "c":
        model = nn.Model_C(ut.T2I, ut.W2I, ut.I2T, ut.P2I, ut.S2I)
    elif repr == "d":
        model = nn.Model_D(ut.T2I, ut.W2I, ut.I2T, ut.C2I)
    else:
        print("Unvalid repr. Program quits")
        sys.exit(1)

    train(model, train_data, dev_data, type)
    #save_model(model, model_file)

if __name__ == "__main__":
    main(*sys.argv[1:])