import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class BOWClassifier(nn.Module):

    def __init__(self, input_size, hidden_size, num_labels):
        torch.backends.cudnn.deterministic = True
        super(BOWClassifier, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.input_size, num_labels)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(x)
        # output = self.sigmoid(output)
        return output


def make_bow_vector(sentence, indexed_vocab, embeddings,input_size):
    # input: sentence which needs to be vectorized, the word to index mapping, word emebeddings
    # output: vector representation for the sentence passed as input
    # calculate the vector of each sentence
    # sum up the vectors of words in the sentence
    # divide the sum vector by the number of words in sentence
    pt_tensor = torch.zeros(input_size, dtype=torch.long)
    count = 0
    for word in sentence:
        count += 1
        if word in indexed_vocab:
            pt_tensor = torch.add(pt_tensor, embeddings(torch.LongTensor([indexed_vocab[word]]))[0])
        else:
            pt_tensor = torch.add(pt_tensor, embeddings(torch.LongTensor([indexed_vocab['#UNK#']]))[0])
    pt_tensor = torch.div(pt_tensor, count)
    return pt_tensor


def get_bow_rep(data, word2idx, embeddings,input_size):
    # controller function to vectorise the sentences. it internally calls function to make bow_vectors
    bow_data = []
    for label, sent in data:
        bow_data.append(make_bow_vector(sent, word2idx, embeddings,input_size))
    return torch.stack(bow_data)


def train_bow_model(data_set, num_unique_labels, word2idx, embeddings, cfg, input_size, use_random):
    # A function to train the bag og words model. All configuration can be controlled from config file-> bow.config
    print("Training started !!! It may take a few minutes.")
    training_set = get_bow_rep(data_set, word2idx, embeddings,input_size)
    if use_random:
        training_set = training_set.clone().detach().requires_grad_(False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # input_size = 300
    hidden_size = cfg['network']['hidden_size']
    num_classes = num_unique_labels
    num_epochs = cfg['hyperparams']['epochs']
    learning_rate = cfg['hyperparams']['learning_rate']
    batch_size = cfg['hyperparams']['batch_size']

    # print (hidden_size,num_classes,num_epochs,learning_rate,batch_size,input_size)

    model = BOWClassifier(input_size, hidden_size, num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # training in batches
    for epoch in range(num_epochs):
        permutation = torch.randperm(training_set.size()[0])
        for i in range(0, training_set.size()[0], batch_size):
            optimizer.zero_grad()
            indices = permutation[i:i + batch_size]
            batch_features = training_set[indices]
            batch_features = batch_features.reshape(-1, input_size).to(device)
            batch_labels = torch.LongTensor([label for label, sent in data_set[indices]]).to(device)
            batch_outputs = model(batch_features)
            loss = criterion(batch_outputs, batch_labels)
            loss.backward()
            optimizer.step()
        # print(loss.item())
        # print(epoch)
    return model


def evaluate_bow(test, model, word2idx, embeddings, idx2word, output_path,input_size):
    # A fucntion to evaluate the BAG of Words model . It also saves the output to ../data/output.txt.
    # post the prediction of label, it calls the function to get the label string from label index.
    predicted_labels = []
    test_len = len(test)
    correct = 0

    for label, data in test:
        bow_vec = make_bow_vector(data, word2idx, embeddings,input_size)
        logprobs = model(bow_vec)
        logprobs = F.softmax(logprobs)
        pred = np.argmax(logprobs.data.numpy())
        actual_label = idx2word.get(pred)
        predicted_labels.append(actual_label)
        if pred == label:
            correct += 1
        # print('prediction: {}'.format(pred))
        # print('actual: {}'.format(label))
    accuracy = correct / test_len
    # print('accuracy: {}'.format(accuracy))

    with open(output_path, 'w') as f:
        for item in predicted_labels:
            f.write("{}\n".format(item))
        f.write('accuracy: {}'.format(accuracy * 100))

    return accuracy
