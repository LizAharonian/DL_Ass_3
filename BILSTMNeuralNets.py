
import dynet as dy
import utils_part_3 as ut
import numpy as np

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
        self.first_forward = dy.LSTMBuilder(1, WORD_EMBEDDING_DIM, LSTM_DIM, self.model)
        self.first_backward = dy.LSTMBuilder(1, WORD_EMBEDDING_DIM, LSTM_DIM, self.model)

        # second BILSTM - input: b1,..bn, output: b'1,..b'n
        self.second_forward = dy.LSTMBuilder(1, WORD_EMBEDDING_DIM, LSTM_DIM, self.model)
        self.second_backward = dy.LSTMBuilder(1, WORD_EMBEDDING_DIM, LSTM_DIM, self.model)

        # MLP mult on: b'1,..b'n
        self.W1 = self.model.add_parameters((MLP_DIM, WORD_EMBEDDING_DIM))
        self.W2 = self.model.add_parameters((ut.T2I.size, MLP_DIM))

    def build_graph(self, sentence):
        dy.renew_cg()
        # initialize the bilstm layers
        self.first_forward_initialize = self.first_forward.initial_state()
        self.first_backward_initialize = self.first_backward.initial_state()
        self.second_forward_initialize = self.second_forward.initial_state()
        self.second_backward_initialize = self.second_backward.initial_state()

        # get the word vectors. word_rep(...) returns a 128-dim vector expression for each word.
        words_embedding_list = [self.E(ut.W2I[word]) for word in sentence]
        reversed_words_embedding_list = list(reversed(words_embedding_list))

        # first BILSTM layer, input: x1,x2,.. xn (word_embeding_list) output: b1,b2,b3,..bn
        forward_y = self.first_forward_initialize .transduce(words_embedding_list)
        backward_y = self.first_backward_initialize .transduce(reversed_words_embedding_list)

        # concat the results
        b = [[y1,y2] for y1,y2 in zip(forward_y, backward_y)]

        # second BILSTM layer, input: b1,b2..bn, output: b'1,b'2, b'3..
        forward_y_tag = self.second_forward_initialize .transduce(b)
        backward_y_tag = self.second_backward_initialize .transduce(list(reversed(b)))

        # concat the results
        b_tag = [[y1_tag,y2_tag] for y1_tag, y2_tag in zip(forward_y_tag,backward_y_tag)]

        # insert b_tag list into MLP
        W1 = dy.parameter(self.W1)
        W2 = dy.parameter(self.W2)

        result = []
        for b_tag_item in b_tag:
            result.append(self.W2 *(dy.tanh(self.W1*b_tag_item)))
        return result

    def get_train_loss(self, sentence,tags):
        """
        get_train_loss function.
        :param sentence: words list of sentence from train file.
        :param tags: the relevant tags of the sentence.
        :return:
        """
        result = self.build_graph()
        loss = 0.0
        for r, tag in zip(result, tags):
            loss += dy.pickneglogsoftmax_batch(r,ut.T2I[tags])
        return loss

    def get_prediction_on_sentence(self, sentence):
        results = self.build_graph(sentence)
        probs = [(dy.softmax(r)).npvalue() for r in results]
        tags = [ut.T2I[np.argmax(pro)] for pro in probs]
        return tags
    


