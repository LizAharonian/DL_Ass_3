UNK = "UUUNKKK"
PREFIX_SIZE = 3
SUFFIX_SIZE = 3
WORDS_SET = set()
TAGS_SET = set()
CHARS_SET = set()
W2I = {}
T2I = {}
C2I = {}
I2W = {}
I2T = {}
I2C = {}
P2I = {}
S2I = {}

def read_tagged_data(file_name, is_dev = False):
    """
    read_tagged_data function.
    reads dev and train from files and returns list of tagged sentences.
    in case we read the train, we also fill the WORDS_SET and TAGS_SET
    :param file_name: name of file to read.
    :param is_dev: indicates if the file is validation file.
    :return: list of tagged sentences
    """
    global WORDS_SET, TAGS_SET, CHARS_SET
    tagged_sentences = []
    with open(file_name) as file:
        content = file.readlines()
        sentence_and_tags = []
        for line in content:
            if line == "\n":
                tagged_sentences.append(sentence_and_tags)
                sentence_and_tags =[]
                continue
            line = line.strip("\n").strip().strip("\t")
            word, tag  = line.split()
            if not is_dev:
                TAGS_SET.add(tag)
                WORDS_SET.add(word)
                CHARS_SET.update(word)
            sentence_and_tags.append((word, tag))
    if not is_dev:
        TAGS_SET.add(UNK)
        WORDS_SET.add(UNK)
    return tagged_sentences

def read_not_tagged_data(file_name):
    """
    read_not_tagged_data function.
    reads the test file.
    :param file_name: test file name.
    :return: list of sentences.
    """
    sentences = []
    with open(file_name) as file:
        content = file.readlines()
        sentence = []
        for line in content:
            if line == "\n":
                sentences.append(sentence)
                sentence =[]
                continue
            w = line.strip("\n").strip()
            sentence.append(w)
    return sentences

def load_indexers():
    """
    load_indexers function.
    creates our dicts that helps us to manage the data.
    """
    global  WORDS_SET, W2I, TAGS_SET, T2I, CHARS_SET, C2I, PREFIX_SIZE, SUFFIX_SIZE, P2I, S2I
    global I2W, I2T, I2C, P2I, S2I
    W2I = {word : i for i, word in enumerate(WORDS_SET)}
    I2W = {i : word for word, i in W2I.iteritems()}
    T2I = {tag : i for i, tag in enumerate(TAGS_SET)}
    I2T = {i : word for word, i in T2I.iteritems()}
    C2I = {tag : i for i, tag in enumerate(CHARS_SET)}
    I2C = {i : word for word, i in C2I.iteritems()}

    # initialize prefixes and suffixes
    P2I = {word[:PREFIX_SIZE]:i for i, word in enumerate(WORDS_SET)}
    S2I = {word[:-SUFFIX_SIZE]:i for i, word in enumerate(WORDS_SET)}