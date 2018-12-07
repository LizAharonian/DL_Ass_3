import dynet as dy
import sys
import numpy as np
import random as rand

EMBED_DIM = 100
LSTM_DIM = 100
OUT_DIM = 2
EPOCHS = 10
VOCAB ="0123456789abcd"
VOCAB_SIZE = len(VOCAB)
V2I = {char: i for i, char in enumerate(VOCAB)}

NUM_LAYERS = 2
EMBED_DIM = 50
VOCAB_SIZE = len(VOCAB)
IN_DIM =  150
HIDE_DIM = 100
OUT_DIM = 2



def main(argv):

    train_file_name = argv[0]
    test_file_name = argv[1]
    train_data = read_file_and_get_data(train_file_name)
    test_data = read_file_and_get_data(test_file_name)
    model = RNNAcceptorModel()
    model.train(train_data,test_data)



    pass


def read_file_and_get_data(file_name):
    with open(file_name) as file:
        lines = file.readlines()
        tagged_examples = []
        for line in lines:
            x, y = line.strip('\n').split(" ")
            tagged_examples.append((x,int(y)))

    return tagged_examples



# acceptor LSTM
class LstmAcceptor(object):

    def __init__(self, in_dim, lstm_dim, out_dim, model):
        self.builder = dy.VanillaLSTMBuilder(1, in_dim, lstm_dim, model)
        self.W       = model.add_parameters((out_dim, lstm_dim))

    def __call__(self, sequence):
        lstm = self.builder.initial_state()
        W = self.W.expr() # convert the parameter into an Expession (add it to graph)
        outputs = lstm.transduce(sequence)
        result = W*outputs[-1]
        return result

# our RNN model
class RNNAcceptorModel(object):




    def __init__(self):
        self.m = dy.Model()
        self.trainer = dy.AdamTrainer(self.m)
        self.embeds = self.m.add_lookup_parameters((VOCAB_SIZE, EMBED_DIM))
        self.acceptor = LstmAcceptor(EMBED_DIM, LSTM_DIM, OUT_DIM, self.m)
        # self.m = dy.ParameterCollection() # initialize the model, holds the parameters
        # self.trainer = dy.AdamTrainer(self.m)
        # self.__builder = dy.LSTMBuilder(NUM_LAYERS, EMBED_DIM, IN_DIM, self.m)
        # self.__E = self.m.add_lookup_parameters((VOCAB_SIZE, EMBED_DIM))
        # self.__W1 = self.m.add_parameters((HIDE_DIM, IN_DIM))
        # self.__b1 = self.m.add_parameters(HIDE_DIM)
        # self.__W2 = self.m.add_parameters((OUT_DIM, HIDE_DIM))
        # self.__b2 = self.m.add_parameters(OUT_DIM)

    def train(self,train_data,test_data):
        # training code
        sum_of_losses = 0.0
        for epoch in range(EPOCHS):
            print "Epoch number " + str(epoch) + " started!"
            # training iter
            rand.shuffle(train_data)
            for x, label in train_data:
                dy.renew_cg()  # new computation graph
                vecs = [self.embeds[V2I[char]] for char in x]
                preds = self.acceptor(vecs)
                loss = dy.pickneglogsoftmax(preds, label)
                sum_of_losses += loss.npvalue()
                loss.backward()
                self.trainer.update()
            # end of one iter on train data
            print "train: " + "loss is: " + str(float(sum_of_losses) / len(train_data)) + " accuracy is: " + str(self.compute_accuracy(train_data))
            self.run_test_and_print_accuracy_and_loss(test_data)
            sum_of_losses = 0.0

    def run_test_and_print_accuracy_and_loss(self,test_data):
        sum_of_losses = 0.0
        for x, label in test_data:
            dy.renew_cg()  # new computation graph
            vecs = [self.embeds[V2I[char]] for char in x]
            preds = self.acceptor(vecs)
            loss = dy.pickneglogsoftmax(preds, label)
            sum_of_losses += loss.npvalue()
            loss.backward()
            # end of one iter on test data
        print "test: " + "loss is: " + str(float(sum_of_losses) / len(test_data)) + " accuracy is: " + str(
            self.compute_accuracy(test_data))

    def predict(self, w):
        # prediction code:
        dy.renew_cg()  # new computation graph
        vecs = [self.embeds[V2I[char]] for char in w]
        preds = dy.softmax(self.acceptor(vecs))
        vals = preds.npvalue()
        return np.argmax(vals)

    def compute_accuracy(self, tagged_data):
        good = bad = 0
        for x, y in tagged_data:
            pred = self.predict(x)
            if pred == y:
                good += 1
            else:
                bad += 1
        return float(good) / float(len(tagged_data))

if __name__ == "__main__":
    main(sys.argv[1:])