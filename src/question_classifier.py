import getopt
import sys

import torch

import yaml
from Util.BOW import *
from Util.LSTM import *
from Util.utils import *


def main(method, cfg):
    if method == 'train':
        # get all the training data for preparaing the label set and store the labels for future use
        complete_data = pre_process(cfg['data']['questions']    , cfg)
        labels = get_labels(complete_data)
        complete_data = append_labels(complete_data, labels)
        store_labels(labels, cfg['data']['labels'])
        num_unique_labels = len(set(labels))

        # preprare the training data
        data_train = cfg['data']['train']
        data_train_preprocessed = pre_process(data_train, cfg)
        data_train_transformed = append_labels(data_train_preprocessed, labels)
        use_random = cfg['use_pretrained']['random']

        # create word2idx , embeddings and vocab for training the model
        word2idx = get_word2idx(use_random, cfg,complete_data)
        embeddings, _glove = get_embeddings(use_random, word2idx, cfg)
        input_size = _glove.size(1)



        if (cfg['model']['name']).lower() == 'bow':

            mod = train_bow_model(data_train_transformed, num_unique_labels, word2idx, embeddings, cfg, input_size,use_random)
            save_model(mod, cfg['data']['model'])

        elif (cfg['model']['name']).lower() == 'bilstm':
            # params for LSTM model
            vocab_size = _glove.size(0)
            embedding_dim = _glove.size(1)
            tagset_size = num_unique_labels
            hidden_dim = embedding_dim
            learning_rate = cfg['hyperparams']['learning_rate']

            # model training
            lstm_model = train_bilstm_model(data_train_transformed, embedding_dim, hidden_dim, vocab_size, tagset_size,
                                            learning_rate,
                                            word2idx, use_random, _glove, cfg)
            save_model(lstm_model, cfg['data']['model'])

    elif method == 'test':

        # load the labels from training data:label is key
        labels = load_labels(cfg['data']['labels'])

        # prepare testing data
        data_test = cfg['data']['test']
        data_test_preprocessed = pre_process(data_test, cfg)
        data_test_transformed = append_labels(data_test_preprocessed, labels)

        # convert index as key for look-up during prediction
        index2label = get_idx2word(labels)

        num_unique_labels = len(set(labels))

        if (cfg['model']['name']).lower() == 'bow':

            # params for BOWModel loader
            input_size = cfg['network']['input_size']
            hidden_size = cfg['network']['hidden_size']
            num_classes = num_unique_labels

            # load the vocab and word2idx
            word2idx = torch.load(cfg['data']['word2idx'])
            embeddings = torch.load(cfg['data']['embeddings'])

            # load saved model and use for interpretation
            model = BOWClassifier(input_size, hidden_size, num_classes)
            model.load_state_dict(torch.load(cfg['data']['model']))
            model.eval()

            # evaluate the model and save the data
            acc = evaluate_bow(data_test_transformed, model, word2idx, embeddings, index2label, cfg['data']['output'])

            print("\nAccuracyBOW=", acc * 100)

        elif (cfg['model']['name']).lower() == 'bilstm':

            # load the vocab and word2idx
            word2idx = torch.load(cfg['data']['word2idx'])
            _glove = torch.load(cfg['data']['vocab'])

            # params for LSTMModel loader
            vocab_size = _glove.size(0)
            embedding_dim = _glove.size(1)
            tagset_size = num_unique_labels
            hidden_dim = embedding_dim
            use_random = cfg['use_pretrained']['random']

            # load saved model and use for interpretation
            model = LSTMTagger(embedding_dim, hidden_dim, vocab_size, tagset_size, use_random, _glove)
            model.load_state_dict(torch.load(cfg['data']['model']))
            model.eval()

            # evaluate the model and save the data
            acc = evaluate_lstm(data_test_transformed, word2idx, model, index2label, cfg['data']['output'])
            print("\nAccuracy=", acc * 100)


    else:
        print("Please check the argument . Expected value train or test")
        sys.exit(1)


if __name__ == "__main__":

    cmd_args = sys.argv
    if len(cmd_args) == 4 and sys.argv[2] in ["-c", "--config", "-config", "config"]:
        method = (sys.argv[1]).lower()
        conf_path = sys.argv[3]
        with open(conf_path, 'r') as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
        torch.backends.cudnn.deterministic = True
        main(method, cfg)
    else:
        print("Seems arguments are wrong..")
        print("usage:: python3 question_classifier.py train/test -config <path>")
        print ("Ex:: python3 question_classifier.py train  -config ../data/bilstm.config")
        sys.exit(2)

    # method = (sys.argv[1]).lower()
    # config_path = ''
    # try:
    #     opts, args = getopt.getopt(sys.argv[2:], "hc:", ["config="])
    # except getopt.GetoptError:
    #     print('question_classifier.py train/test --config <path>')
    #     sys.exit(2)
    # for opt, arg in opts:
    #     if opt == '-h':
    #         print('question_classifier.py train/test --config <path>')
    #         sys.exit()
    #     elif opt in ('-c', '--config'):
    #         config_path = arg
    #
    # with open(config_path, 'r') as ymlfile:
    #     cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    # torch.backends.cudnn.deterministic = True
    # # main(method, cfg)
