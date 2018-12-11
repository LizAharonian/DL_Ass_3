import sys
import cPickle as pickle
from zipfile import ZipFile


def load_dicts_from_modelFile(modelFile):
    dicts = []
    with ZipFile("modelFile") as myzip:
        with myzip.open("dicts.pkl") as dicts_file:
            W2I = pickle.load(dicts_file)
            T2I = pickle.load(dicts_file)
            C2I = pickle.load(dicts_file)
            I2W = pickle.load(dicts_file)
            I2T = pickle.load(dicts_file)
            I2C = pickle.load(dicts_file)
            P2I = pickle.load(dicts_file)
            S2I = pickle.load(dicts_file)
            dicts += [W2I, T2I, C2I, I2W, I2T, ]

def main(repr, modelFile, inputFile):
    load_modelFile(modelFile)


if __name__ == "__main__":
    main(*sys.argv[1:])


