import cPickle as pickle

with open('modelFile', 'rb') as input:
    W2I = pickle.load(input)
    T2I = pickle.load(input)

    x = 5