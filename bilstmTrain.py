import sys
import utils_part_3 as ut

def main(repr, train_file, model_file, dev_file=None):
    train_data = ut.read_tagged_data(train_file)
    if dev_file:
        dev_data = ut.read_tagged_data(dev_file)
    else:
        data_len = len()
        dev_data = train_file[]



if __name__ == "__main__":
    main(*sys.argv[1:])