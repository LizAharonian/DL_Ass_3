import sys
import os
import cPickle as pickle
import BILSTMNeuralNets as nn
from zipfile import ZipFile
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D


def load_dicts_from_modelFile(pkl_name):
    with open(pkl_name) as dicts_file:
        dict = pickle.load(dicts_file)
    return dict

a_pos = load_dicts_from_modelFile("a_model_pos.pkl")
b_pos = load_dicts_from_modelFile("b_model_pos.pkl")
c_pos = load_dicts_from_modelFile("c_model_pos.pkl")
d_pos = load_dicts_from_modelFile("d_model_pos.pkl")

a_ner = load_dicts_from_modelFile("a_model_ner.pkl")
b_ner = load_dicts_from_modelFile("b_model_ner.pkl")
c_ner = load_dicts_from_modelFile("c_model_ner.pkl")
d_ner = load_dicts_from_modelFile("d_model_ner.pkl")

label1, = plt.plot(a_pos.keys(), a_pos.values(), "b-", label='model a pos')
label2, = plt.plot(b_pos.keys(), b_pos.values(), "r-", label='model b pos')
label3, = plt.plot(c_pos.keys(), c_pos.values(), "g-", label='model c pos')
label4, = plt.plot(d_pos.keys(), d_pos.values(), "y-", label='model d pos')
plt.legend(handler_map={label1: HandlerLine2D(numpoints=4)})
plt.show()

label1, = plt.plot(a_ner.keys(), a_pos.values(), "b-", label='model a ner')
label2, = plt.plot(b_ner.keys(), b_pos.values(), "r-", label='model b ner')
label3, = plt.plot(c_ner.keys(), c_pos.values(), "g-", label='model c ner')
label4, = plt.plot(d_ner.keys(), d_pos.values(), "y-", label='model d ner')
plt.legend(handler_map={label1: HandlerLine2D(numpoints=4)})
plt.show()
