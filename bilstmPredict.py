import sys
import os
import cPickle as pickle
import BILSTMNeuralNets as nn
from zipfile import ZipFile
import utils_part_3 as ut

def load_dicts_from_modelFile():
    dicts = []
    with open("dicts.pkl") as dicts_file:
        W2I = pickle.load(dicts_file)
        T2I = pickle.load(dicts_file)
        C2I = pickle.load(dicts_file)
        I2W = pickle.load(dicts_file)
        I2T = pickle.load(dicts_file)
        I2C = pickle.load(dicts_file)
        P2I = pickle.load(dicts_file)
        S2I = pickle.load(dicts_file)
        dicts += [W2I, T2I, C2I, I2W, I2T, I2C, P2I, S2I]
    return dicts

def load_model(model):
        model.model.populate("model.dy")

def get_test_pred(test_data, model):
    pred_list = []
    for sentence in test_data:
        pred = model.get_prediction_on_sentence(sentence)
        pred_list.append((sentence,pred))
    return  pred_list

def write_test_pred(preds_list, data_type):
    with open("test4."+data_type, 'w') as test_pred_file:
        for item in preds_list:
            sentence, tags = item
            for w,tag in zip(sentence,tags):
                test_pred_file.write(w + " " + tag + "\n")
            test_pred_file.write("\n")

def main(repr, modelFile, inputFile, type):
    with ZipFile(modelFile) as myzip:
        myzip.extractall(os.getcwd())
    W2I, T2I, C2I, I2W, I2T, I2C, P2I, S2I = load_dicts_from_modelFile()
    # Initialize model
    if repr == "a":
        model = nn.Model_A(T2I, W2I, I2T)
    elif repr == "b":
        model = nn.Model_B(T2I, W2I, I2T, C2I)
    elif repr == "c":
        model = nn.Model_C(T2I, W2I, I2T, P2I, S2I)
    elif repr == "d":
        model = nn.Model_D(T2I, W2I, I2T, C2I)
    else:
        print("Unvalid repr. Program quits")
        sys.exit(1)

    load_model(model)
    os.remove("dicts.pkl")
    os.remove("model.dy")
    test_data = ut.read_not_tagged_data(inputFile)
    preds_list = get_test_pred(test_data, model)
    write_test_pred(preds_list, type)

if __name__ == "__main__":
    main(*sys.argv[1:])

