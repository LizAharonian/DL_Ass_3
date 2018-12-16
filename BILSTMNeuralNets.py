
import dynet as dy
from utils_part_3 import UNK
import numpy as np

STUDENT = {'name': "Liz Aharonian_Ori Ben Zaken",
           'ID': "316584960_311492110"}

WORD_EMBEDDING_DIM = 128
MLP_DIM = 32
LSTM_DIM = 64
# for B model
CHAR_EMBEDDING_DIM = 20
CHAR_LSTM_DIM = 128

# for model C
PREF_EMBEDDING_DIM = 128
SUFF_EMBEDDING_DIM = 128




class Model_A(object):
    def __init__(self, rep, T2I, W2I,I2T):
        """
        constructor.
        the params are all dics from the utils.
        :param T2I: 
        :param W2I: 
        :param I2T: 
        """
        self.T2I = T2I
        self.W2I = W2I
        self.I2T = I2T
        self.model = dy.ParameterCollection()
        if rep == "d":
            lr = 0.00014
        else:
            lr = 0.00045
        self.trainer = dy.AdamTrainer(self.model,lr)
        # word embedding matrix
        self.E = self.model.add_lookup_parameters((len(self.W2I), WORD_EMBEDDING_DIM))

        # first BILSTM - input: x1,..xn, output: b1,..bn
        self.first_forward = dy.LSTMBuilder(1, WORD_EMBEDDING_DIM, LSTM_DIM, self.model)
        self.first_backward = dy.LSTMBuilder(1, WORD_EMBEDDING_DIM, LSTM_DIM, self.model)

        # second BILSTM - input: b1,..bn, output: b'1,..b'n
        self.second_forward = dy.LSTMBuilder(1, WORD_EMBEDDING_DIM, LSTM_DIM, self.model)
        self.second_backward = dy.LSTMBuilder(1, WORD_EMBEDDING_DIM, LSTM_DIM, self.model)

        # MLP mult on: b'1,..b'n
        self.W1 = self.model.add_parameters((MLP_DIM, WORD_EMBEDDING_DIM))
        self.W2 = self.model.add_parameters((len(self.T2I), MLP_DIM))

    def build_graph(self, sentence):
        """
        build_graph function.
        :param sentence: input sentence.
        :return: 
        """
        dy.renew_cg()
        # initialize the bilstm layers
        self.first_forward_initialize = self.first_forward.initial_state()
        self.first_backward_initialize = self.first_backward.initial_state()
        self.second_forward_initialize = self.second_forward.initial_state()
        self.second_backward_initialize = self.second_backward.initial_state()

        # get the word vectors. word_rep(...) returns a 128-dim vector expression for each word.
        words_embedding_list = [self.get_word_rep(word) for word in sentence]
        reversed_words_embedding_list = reversed(words_embedding_list)

        # first BILSTM layer, input: x1,x2,.. xn (word_embeding_list) output: b1,b2,b3,..bn
        forward_y = self.first_forward_initialize.transduce(words_embedding_list)
        backward_y = self.first_backward_initialize.transduce(reversed_words_embedding_list)

        # concat the results
        b = [dy.concatenate([y1,y2]) for y1,y2 in zip(forward_y, backward_y)]

        # second BILSTM layer, input: b1,b2..bn, output: b'1,b'2, b'3..
        forward_y_tag = self.second_forward_initialize.transduce(b)
        backward_y_tag = self.second_backward_initialize.transduce(reversed(b))

        # concat the results
        b_tag = [dy.concatenate([y1_tag,y2_tag]) for y1_tag, y2_tag in zip(forward_y_tag,backward_y_tag)]

        # insert b_tag list into MLP
        W1 = dy.parameter(self.W1)
        W2 = dy.parameter(self.W2)

        result = []
        for b_tag_item in b_tag:
            result.append(W2 *(dy.tanh(W1*b_tag_item)))
        return result

    def get_word_rep(self,word):
        """
        get_word_rep function.
        :param word: requested word.
        :return: 
        """
        if word in self.W2I.keys():
            return self.E[self.W2I[word]]
        else:
            return self.E[self.W2I[UNK]]

    def get_train_loss(self, sentence,tags):
        """
        get_train_loss function.
        :param sentence: words list of sentence from train file.
        :param tags: the relevant tags of the sentence.
        :return:
        """
        result = self.build_graph(sentence)
        loss = []
        for r, tag in zip(result, tags):
            loss.append(dy.pickneglogsoftmax(r,self.T2I[tag]))
        return dy.esum(loss)

    def get_prediction_on_sentence(self, sentence):
        """
        get_prediction_on_sentence function.
        :param sentence: sentence to be tagged.
        :return: 
        """
        results = self.build_graph(sentence)
        probs = [(dy.softmax(r)).npvalue() for r in results]
        tags = [self.I2T[np.argmax(pro)] for pro in probs]
        return tags

    
class Model_B(Model_A):
    def __init__(self,rep,T2I, W2I,I2T,C2I):
        """
        constructor.
        all params are dicts from utils.
        :param T2I: 
        :param W2I: 
        :param I2T: 
        :param C2I: 
        """
        self.C2I = C2I
        super(Model_B, self).__init__(rep,T2I,W2I,I2T)
        self.E_CHAR = self.model.add_lookup_parameters((len(C2I), CHAR_EMBEDDING_DIM))
        self.char_LSTM = dy.LSTMBuilder(1, CHAR_EMBEDDING_DIM, CHAR_LSTM_DIM, self.model)

    def get_word_rep(self,word):
        """
        get_word_rep function.
        :param word: requested word.
        :return: 
        """
        char_indexes = []
        for char in word:
            if char in self.C2I:
                char_indexes.append(self.C2I[char])
            else:
                char_indexes.append(self.C2I[UNK])
        char_embedding = [self.E_CHAR[indx] for indx in char_indexes]
        char_lstm_init = self.char_LSTM.initial_state()
        # calculate y1,y2,..yn and return yn
        return char_lstm_init.transduce(char_embedding)[-1]

class Model_C(Model_A):
    def __init__(self ,rep,T2I, W2I,I2T, P2I, STI):
        """
        constructor.
        all dicts are from utils.
        :param T2I: 
        :param W2I: 
        :param I2T: 
        :param P2I: 
        :param STI: 
        """
        super(Model_C, self).__init__(rep,T2I,W2I,I2T)
        self.P2I = P2I
        self.STI = STI
        self.E_PREF = self.model.add_lookup_parameters((len(self.P2I), PREF_EMBEDDING_DIM))
        self.E_SUFF = self.model.add_lookup_parameters((len(self.STI), SUFF_EMBEDDING_DIM))

    def get_word_rep(self, word):
        """
        get_word_rep function.
        :param word: requested word.
        :return:
        """
        prefix = word[:3]
        suffix = word[-3:]

        if prefix in self.P2I.keys():
            pref_indx = self.P2I[prefix]
        else:
            pref_indx = self.P2I[UNK[:3]]

        if suffix in self.STI.keys():
            suff_indx = self.STI[suffix]
        else:
            suff_indx = self.STI[UNK[-3:]]

        return dy.esum([self.E_PREF[pref_indx], self.E_SUFF[suff_indx]])

class Model_D(Model_B):
    def __init__(self,rep,T2I, W2I,I2T, C2I):
        """
        constructor.
        all dicts are from utils.
        :param T2I:
        :param W2I:
        :param I2T:
        :param C2I:
        """
        super(Model_D, self).__init__(rep,T2I, W2I,I2T,C2I)
        #params for linear layer
        self.W = self.model.add_parameters((WORD_EMBEDDING_DIM, WORD_EMBEDDING_DIM * 2))
        self.b = self.model.add_parameters((WORD_EMBEDDING_DIM))

    def get_word_rep(self, word):
        """
        get_word_rep function.
        :param word: requested word.
        :return:
        """
        # making params for linear layer
        W = dy.parameter(self.W)
        b = dy.parameter(self.b)

        word_embeddings_from_a_model = Model_A.get_word_rep(self, word)
        word_embeddings_from_b_model = Model_B.get_word_rep(self, word)

        word_embeddings_d_model = dy.concatenate([word_embeddings_from_a_model, word_embeddings_from_b_model])

        # linear layer calculations
        res = ((W * word_embeddings_d_model) + b)
        return res