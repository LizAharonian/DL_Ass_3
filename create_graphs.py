import cPickle as pickle
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import collections as liz_the_bitch


STUDENT = {'name': "Liz Aharonian_Ori Ben Zaken",
           'ID': "316584960_311492110"}

def load_dicts_from_modelFile(pkl_name):
    with open(pkl_name) as dicts_file:
        dict = pickle.load(dicts_file)
    return liz_the_bitch.OrderedDict(sorted(dict.items()))

a_pos = load_dicts_from_modelFile("folder_a/a_graph_pos")
b_pos = load_dicts_from_modelFile("folder_a/b_graph_pos")
c_pos = load_dicts_from_modelFile("folder_a/c_graph_pos")
d_pos = load_dicts_from_modelFile("folder_a/d_graph_pos")


a_ner = load_dicts_from_modelFile("folder_a/a_graph_ner")
b_ner = load_dicts_from_modelFile("folder_a/b_graph_ner")
c_ner = load_dicts_from_modelFile("folder_a/c_graph_ner")
d_ner = load_dicts_from_modelFile("folder_a/d_graph_ner")

label1, = plt.plot(a_pos.keys(), a_pos.values(), "b-", label='a - pos')
label2, = plt.plot(b_pos.keys(), b_pos.values(), "g-", label='b - pos')
label3, = plt.plot(c_pos.keys(), c_pos.values(), "r-", label='c - pos')
label4, = plt.plot(d_pos.keys(), d_pos.values(), "k-", label='d - pos')
plt.legend(handler_map={label1: HandlerLine2D(numpoints=4)})
plt.ylabel("accuracy")
plt.xlabel("iter number / 100")
plt.show()

label1, = plt.plot(a_ner.keys(), a_ner.values(), "b-", label='a - ner')
label2, = plt.plot(b_ner.keys(), b_ner.values(), "g-", label='b - ner')
label3, = plt.plot(c_ner.keys(), c_ner.values(), "r-", label='c - ner')
label4, = plt.plot(d_ner.keys(), d_ner.values(), "k-", label='d - ner')
plt.legend(handler_map={label1: HandlerLine2D(numpoints=4)})
plt.ylabel("accuracy")
plt.xlabel("iter number / 100")
plt.show()