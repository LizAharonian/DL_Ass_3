import rstr
from random import shuffle

STUDENT = {'name': "Liz Aharonian_Ori Ben Zaken",
           'ID': "316584960_311492110"}

def main():
    """
    main function, runs the program and creates the requested examples for part 2.
    :return:
    """
    # w#w_reverse
    index = "1"
    write_examples_to_file_from_regex(r'[a-z]+',"pos_examples" + index, index)
    write_examples_to_file_from_regex(r'[a-z]+#[a-z]+',"neg_examples" + index, index)
    create_test_and_train(index)

    #w#W
    index = "2"
    write_examples_to_file_from_regex(r'[a-z]{1,5000}', "pos_examples" + index, index)
    write_examples_to_file_from_regex(r'[a-z]{1,5000}#[a-z]{1,5000}', "neg_examples" + index, index)
    create_test_and_train(index)

    # original language from part1
    index = "3"
    write_examples_to_file_from_regex(r'[1-9]+a+[1-9]+b+[1-9]+c+[1-9]+d+[1-9]+', "pos_examples" + index, index)
    write_examples_to_file_from_regex(r'[1-9]+a+[1-9]+c+[1-9]+b+[1-9]+d+[1-9]+', "neg_examples" + index, index)
    create_test_and_train(index)



def write_examples_to_file_from_regex(regex, file_name,mission):
    """
    write the regex to file.
    :param regex: regular expression.
    :param file_name: file name for saving.
    :param mission: 1,2,3
    :return:
    """
    with open(file_name,"w") as file:
        random_examples_list = []
        for i in range(500):
            w = rstr.xeger(regex)
            if mission == "2" and file_name.startswith("neg_examples"):
                random_examples_list.append(w)
                continue
            w_revers = w[::-1]
            w_bkp = w
            if mission == "1" or mission == "2":
                w += "#"
            if mission == "1":
                w += w_revers
            if mission == "2":
                w += w_bkp
            random_examples_list.append(w)
        content = "\n".join(random_examples_list)
        file.write(content)

def create_test_and_train(index):
    """
    create_test_and_train function.
    splites the data.
    :param index: 1,2,3
    :return:
    """
    examples_and_tags_list = []
    with open("neg_examples" + index,"r") as neg_file, open("pos_examples" + index,"r") as pos_file:
        pos_content, neg_content = pos_file.readlines(),neg_file.readlines()
        pos_content = [example.strip('\n') + " 1" for example in pos_content]
        neg_content = [example.strip('\n') + " 0" for example in neg_content]
        examples_and_tags_list += pos_content + neg_content
    shuffle(examples_and_tags_list)
    test_list, train_list  = examples_and_tags_list[:200], examples_and_tags_list[200:]
    with open("train" +index,"w") as train_file, open("test" +index,"w") as test_file:
        train_file.write("\n".join(train_list))
        test_file.write("\n".join(test_list))






if __name__ == "__main__":
    main()
