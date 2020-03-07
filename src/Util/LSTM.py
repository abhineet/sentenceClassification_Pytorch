import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, use_random, glove):
        torch.backends.cudnn.deterministic = True
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        if use_random:
            self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        else:
            self.word_embeddings = nn.Embedding.from_pretrained(glove, freeze=True)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        # Decode the hidden state of the last time step
        tag_space = self.hidden2tag(lstm_out[-1, :])
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


def train_bilstm_model(data, embedding_dim, hidden_dim, vocab_size, tagset_size, learning_rate, word2idx, use_random,
                       glove, cfg):
    # A function to train the BiLSTM  model. All configuration can be controlled from config file-> bilstm.config

    print("Training started !!! It may take a few minutes.")

    ModelLSTM = LSTMTagger(embedding_dim, hidden_dim, vocab_size, tagset_size, use_random, glove)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(ModelLSTM.parameters(), lr=learning_rate)
    num_epochs = cfg['hyperparams']['epochs']
    unk = word2idx.get('#UNK#')
    for epoch in range(num_epochs):
        for label, sent in data:
            ids = []
            for word in sent:
                index = word2idx.get(word)
                if index:
                    ids.append([word2idx[word]])
                else:
                    ids.append([unk])
            batch_labels = torch.tensor([label])
            sentence = torch.tensor(ids).squeeze()
            optimizer.zero_grad()
            batch_outputs = ModelLSTM(sentence)
            loss = criterion(batch_outputs, batch_labels)
            loss.backward()
            optimizer.step()
        #     print("loss", loss.item())
        # print("epoch", epoch)
    return ModelLSTM


def evaluate_lstm(data, word2idx, model, idx2word, output_path):
    # A fucntion to evaluate the BAG of Words model . It also saves the output to ../data/output.txt.
    # post the prediction of label, it calls the function to get the label string from label index.

    predicted_labels = []
    test_len = len(data)
    unk = word2idx.get('#UNK#')
    with torch.no_grad():
        correct = 0
        for label, sent in data:
            ids = []
            for word in sent:
                index = word2idx.get(word)
                if index:
                    ids.append([word2idx[word]])
                else:
                    ids.append([unk])
            sentence = torch.tensor(ids).squeeze()
            batch_outputs = model(sentence)
            predicted = np.argmax(F.softmax(batch_outputs).data.numpy())
            actual_label = idx2word.get(predicted)
            predicted_labels.append(actual_label)
            if predicted == label:
                correct += 1
            # print('prediction: {}'.format(predicted))
            # print('actual: {}'.format(label))
        accuracy = correct / test_len
        # print('accuracy: {}'.format(accuracy))
        with open(output_path, 'w') as f:
            for item in predicted_labels:
                f.write("{}\n".format(item))
            f.write('accuracy: {}'.format(accuracy * 100))
        return accuracy
