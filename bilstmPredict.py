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
            W2I = pickle.load(dicts_file)
            PREFIXES = pickle.load(dicts_file)
            SUFFIXES = pickle.load(dicts_file)

        with myzip.open("model.dy") as model_file:
            pass

def main(repr, modelFile, inputFile):
    load_modelFile(modelFile)


if __name__ == "__main__":
    main(*sys.argv[1:])


