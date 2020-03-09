import copy
import pickle
from collections import Counter
import torch
from Util.BOW import *


def read_data(path):
    # Input: Path to data containing training set
    # Output: list of sentences with labels and sentence token
    # Processing:
    # read the data , tokenise it and lower the token
    # split the data , into labels and sentence token

    f = open(path, "r")
    lines = f.readlines()
    f.close()
    data = []
    for l in lines:
        labelSplit = l.replace('\n', '').split(' ', 1)
        data.append([labelSplit[0], [word.lower() for word in labelSplit[1].split()]])
    return data


def remove_stop_words(data, path):
    # Input data: [[DESC:manner, list(['how', 'serfdom', 'develop', 'leave', 'russia', '?'])]]
    # path : path to stop_word.txt
    # Output: [[DESC:manner, list([, 'serfdom', 'develop', 'leave', 'russia', '?'])]]
    # Processing : removes stop word

    with open(path) as f:
        stop_words = [word for line in f for word in line.split(",")]
    data_without_stop_words = []
    for k, v in data:
        words = [t for t in v if t not in stop_words]
        data_without_stop_words.append((k, words))
    return data_without_stop_words


def get_labels(data):
    # Return a dict of label and index value
    # Output:{'LOC:state': 0,'HUM:title': 1,'LOC:other': 2,'DESC:manner': 3}
    _labels = []
    for k, v in data:
        _labels.append(k)
    _unique_label = list(set(_labels))
    _unique_label_dict = {}
    for k, v in enumerate(_unique_label):
        _unique_label_dict[v] = k

    return _unique_label_dict


def store_labels(labels, path):
    # store the labels dict as pkl to be used during testing
    with open(path, 'wb') as handle:
        pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_labels(path):
    # function to retreive the stored label during training
    with open(path, 'rb') as handle:
        b = pickle.load(handle)
        return b


def append_labels(data, labels):
    # function to convert labels from string to index
    # Input :[[DESC:manner, list(['how', 'serfdom', 'develop', 'leave', 'russia', '?'])]]
    # output: [[3, list(['how', 'serfdom', 'develop', 'leave', 'russia', '?'])]]
    cleaned_data = []
    for k, v in data:
        cleaned_data.append((labels[k], v))

    return np.array(cleaned_data)


def create_indexed_vocab(data):
    # this function assign a key to each word in the sentence only if the count of that word is gt eq 2. i.e it
    # removes the word with lesser frequency
    # Input format : data is in form  [[3, list(['how', 'serfdom', 'develop',
    # 'leave', 'russia', '?'])]]
    # Output format : {{'child': 0,'jaco': 1, 'tics': 2, 'restore': 3, 'inspiration': 4,}

    vocab = []
    for _, sent in data:
        for word in sent:
            vocab.append(word)
    count = Counter(vocab)
    count = {w: count[w] for w in count if count[w] >= 2}
    vocab = []
    for k, v in count.items():
        vocab.append(k)
    indexed_vocab = {word: idx for idx, word in enumerate(vocab)}

    return indexed_vocab


def create_vocab(data):
    # this function assign a key to each word in the sentence.
    # Input format : data is in form  [[3, list(['how', 'serfdom', 'develop', 'leave', 'russia', '?'])]]
    # Output format : {{'child': 0,'jaco': 1, 'tics': 2, 'restore': 3, 'inspiration': 4,}

    total_words_orig = []
    for k, sent in data:
        for word in sent:
            total_words_orig.append(word)
    total_words = list(set(total_words_orig))
    # total_words_str = ' '.join(total_words)
    # vocab = set(total_words_str.split())
    word2idx = {word: idx for idx, word in enumerate(total_words)}  # create word index
    return word2idx


def load_glove_embeddings(path, indexed_vocab, embedding_dim=300):
    # this functions add the embeddings to the 'glove'.
    # It takes the glove , word dict and dimensions as input and returns
    # the embeddings with the new words to train
    with open(path) as f:
        embeddings = np.zeros((len(indexed_vocab), embedding_dim))
        for line in f.readlines():
            values = line.split()
            word = values[0]
            index = indexed_vocab.get(word)
            if index:
                vector = np.array(values[1:], dtype='float32')
                embeddings[index] = vector
        return torch.from_numpy(embeddings).float()


def split_train_test(data, test_ratio):
    #  A function to randomly split the data in train/test set , given a split ratio.
    data_copy = copy.deepcopy(data)
    # np.random.seed(400)
    np.random.shuffle(data_copy)
    test_set_size = int(len(data) * test_ratio)
    test = data_copy[:test_set_size]
    train = data_copy[test_set_size:]
    return train, test


def pre_process(data, cfg):
    # A function to call other functions to preprocess and clean the data , i.e
    # 1) read the data and tokenise
    # 2) remove stop words
    # 3) lower the tokens
    stop_words_path = cfg['data']['stop_words']
    raw_data = read_data(data)
    cleaned_data = remove_stop_words(raw_data, stop_words_path)
    return cleaned_data


def get_word2idx(use_random, cfg,data):
    # A contrlloer function to retreive word2idx to handle cases for using random or pretrained method
    # Internally it calls methods to create vocab
    # data = pre_process(cfg['data']['questions'], cfg)
    # labels = get_labels(data)
    # data = append_labels(data, labels)
    # print ('get_word2idx:::::use_random::::::::::::::',use_random)
    if use_random:
        word2idx = create_indexed_vocab(data)
    else:
        word2idx = create_vocab(data)
    return word2idx


def get_embeddings(use_random, word2idx, cfg):
    # A controller function to retreive embeddings to handle cases to use random or pretrained .
    # It internally calls other function to get embeddings , vocab and word2idx.
    # And saves the embeddings , vocab and word2idx for testing .
    glove_embeddings = cfg['data']['glove']
    word2idx['#UNK#'] = len(word2idx)
    glove = load_glove_embeddings(glove_embeddings, word2idx)
    if use_random :
        embeddings = nn.Embedding(glove.size(0), glove.size(1))
    else:
        embeddings = nn.Embedding.from_pretrained(glove, freeze=True)

    torch.save(embeddings, cfg['data']['embeddings'])
    torch.save(word2idx, cfg['data']['word2idx'])
    torch.save(glove, cfg['data']['vocab'])

    return embeddings, glove


def save_model(model, path):
    # given a model , and path ; save the model at path with pth extension
    torch.save(model.state_dict(), path)
    print('saving model at :::', path)


def get_idx2word(word2idx):
    # function to reverse key,value to value,key for a dictionary
    idx2word = {y: x for x, y in word2idx.items()}
    return idx2word
