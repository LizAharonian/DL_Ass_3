import rstr
from random import shuffle

def main():
    # write_examples_to_file_from_regex(r'[1-9]+a+[1-9]+b+[1-9]+c+[1-9]+d+[1-9]+',
    #                                   "pos_examples")
    write_examples_to_file_from_regex(r'[a-z]+',"pos_examples1")
    write_examples_to_file_from_regex(r'[a-z]+#[a-z]+',"neg_examples1")
    create_test_and_train()


def write_examples_to_file_from_regex(regex, file_name):
    with open(file_name,"w") as file:
        random_examples_list = []
        for i in range(500):
            w = rstr.xeger(regex)
            w_revers = w[::-1]
            w += "#"
            w += w_revers
            random_examples_list.append(w)
        content = "\n".join(random_examples_list)
        file.write(content)

def create_test_and_train():
    examples_and_tags_list = []
    with open("neg_examples1","r") as neg_file, open("pos_examples1","r") as pos_file:
        pos_content, neg_content = pos_file.readlines(),neg_file.readlines()
        pos_content = [example.strip('\n') + " 1" for example in pos_content]
        neg_content = [example.strip('\n') + " 0" for example in neg_content]
        examples_and_tags_list += pos_content + neg_content
    shuffle(examples_and_tags_list)
    test_list, train_list  = examples_and_tags_list[:200], examples_and_tags_list[200:]
    with open("train1","w") as train_file, open("test1","w") as test_file:
        train_file.write("\n".join(train_list))
        test_file.write("\n".join(test_list))






if __name__ == "__main__":
    main()
