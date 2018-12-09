import sys
import utils_part_3 as ut
import BILSTMNeuralNets as nn

def main(repr, train_file, model_file, dev_file=None):
    train_data = ut.read_tagged_data(train_file)
    if dev_file:
        dev_data = ut.read_tagged_data(dev_file)
    else:
        eighty_prec_len = int(len(train_data) * 0.8)
        train_data, dev_data = train_data[:eighty_prec_len], train_data[eighty_prec_len:]

    # Initialize model
    if repr == "a":
        model = nn.Model_A()
    elif repr == "b":
        model = nn.Model_B()
    elif repr == "c":
        model = nn.Model_C()
    elif repr == "d":
        model = nn.Model_D()
    else:
        print("Unvalid repr. Program quits")
        sys.exit(1)




if __name__ == "__main__":
    main(*sys.argv[1:])