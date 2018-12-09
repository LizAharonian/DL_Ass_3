
import dynet as dy
import utils_part_3 as ut

WORD_EMBEDDING_DIM = 100
MLP_DIM = 40
LSTM_DIM = 70


class Model_A(object):
    def __init__(self):
        self.model = dy.ParameterCollection()
        self.trainer = dy.AdamTrainer(self.model)
        # word embedding matrix
        self.E = self.model.add_lookup_parameters((ut.W2I.size, WORD_EMBEDDING_DIM))


        # first BILSTM - input: x1,..xn, output: b1,..bn
        self.fwd1 = dy.LSTMBuilder(1, WORD_EMBEDDING_DIM, LSTM_DIM, self.model)
        self.bwd1 = dy.LSTMBuilder(1, WORD_EMBEDDING_DIM, LSTM_DIM, self.model)

        # second BILSTM - input: b1,..bn, output: b'1,..b'n
        self.fwd2 = dy.LSTMBuilder(1, WORD_EMBEDDING_DIM, LSTM_DIM, self.model)
        self.bwd2 = dy.LSTMBuilder(1, WORD_EMBEDDING_DIM, LSTM_DIM, self.model)

        # MLP mult on: b'1,..b'n
        self.W1 = self.model.add_parameters((MLP_DIM, WORD_EMBEDDING_DIM))
        self.W2 = self.model.add_parameters((ut.T2I.size, MLP_DIM))

    def build_tagging_graph(self, words):
        dy.renew_cg()
        W1 = dy.parameter(self.W1)
        W2 = dy.parameter(self.W2)

        # get the word vectors. word_rep(...) returns a 128-dim vector expression for each word.
        words_embedding_list = [self.E(ut.W2I[word]) for word in words]

