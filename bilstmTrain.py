import sys
import time
import random as rand
import dynet as dy
import utils_part_3 as ut
import BILSTMNeuralNets as nn

EPOCHS = 5
TRIANED_EXAMPLES_UNTIL_DEV = 500


def compute_accuracy(self, tagged_data):
    good = bad = 0
    for x, y in tagged_data:
        pred = self.predict(x)
        if pred == y:
            good += 1
        else:
            bad += 1
    return float(good) / float(len(tagged_data))

def split_sentence_to_words_and_tags(tagged_sentence):
    words = [word for word, tag in tagged_sentence]
    tags = [tag for word, tag in tagged_sentence]
    return words, tags

def dev(model, dev_data):
    sum_of_losses = 0.0
    for tagged_sentence in dev_data:
        words, tags = split_sentence_to_words_and_tags(words, tags)
        loss = model.get_train_loss()
        sum_of_losses += loss.npvalue()
        loss.backward()
        # end of one iter on test data
    print "dev results: " + "loss is: " + str(float(sum_of_losses) / len(dev_data)) + " accuracy is: " + \
          str(compute_accuracy(dev_data))

def train(model, train_data, dev_data):
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
                dev(model, dev_data)
            i += 1

        # end of one iter on train data
        print "train epoch" + epoch + "results: " + "loss is: " + str(float(sum_of_losses) / len(train_data)) + \
              " accuracy is: " + str(compute_accuracy(train_data))
        sum_of_losses = 0.0
    end_time = time()
    total_time = end_time - start_time
    print "total time: " + str(total_time)

def save_model(model, model_file):
    model.model.save(model_file)

def main(repr, train_file, model_file, dev_file=None):
    train_data = ut.read_tagged_data(train_file)
    if dev_file:
        dev_data = ut.read_tagged_data(dev_file)
    else:
        eighty_prec_len = int(len(train_data) * 0.8)
        train_data, dev_data = train_data[:eighty_prec_len], train_data[eighty_prec_len:]

    # Initialize model
    if repr == "a":
        model = nn.Model_A()
    elif repr == "b":
        model = nn.Model_B()
    elif repr == "c":
        model = nn.Model_C()
    elif repr == "d":
        model = nn.Model_D()
    else:
        print("Unvalid repr. Program quits")
        sys.exit(1)

    train(model, train_data, dev_data)
    save_model(model, model_file)

if __name__ == "__main__":
    main(*sys.argv[1:])