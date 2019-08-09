# Author: Robert Guthrie

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]
word_to_ix = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
print(word_to_ix)
tag_to_ix = {"DET": 0, "NN": 1, "V": 2}

# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
EMBEDDING_DIM = 6
HIDDEN_DIM = 6

class CharEmbLSTMTagger(nn.Module):

    def __init__(self, embedding_dim, char_embedding_dim, hidden_dim, charset_size, vocab_size, tagset_size):
        super(CharEmbLSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.char_embeddings = nn.Embedding(charset_size, char_embedding_dim)
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm_char = nn.LSTM(char_embedding_dim, char_embedding_dim)
        self.lstm_word = nn.LSTM(embedding_dim + char_embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, char_sentence, sentence):
        word_char_embeds = []
        for i in range(len(char_sentence)):
            char_embeds = self.char_embeddings(char_sentence[i])
            char_lstm_out, hidden_w = self.lstm_char(char_embeds.view(len(char_sentence[i]),1,-1))
            word_embeds = self.word_embeddings(sentence[i])
            word_char_embeds.append(torch.cat((word_embeds, hidden_w[0].view(-1)), dim=0))

        word_char_embeds = torch.stack(word_char_embeds)
        lstm_out, _ = self.lstm_word(word_char_embeds.view(len(sentence), 1, -1))

        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

    # def feed_char(self, char_word):
    #
"""
<class 'list'>: [[Parameter containing:
tensor([[ 0.2322,  0.1555],
        [ 0.2571,  0.3505],
        [-0.6549,  0.3559],
        [-0.4972, -0.5335],     ==> weşght_ih_I0
        [ 0.0430, -0.1205],
        [ 0.4153, -0.4095],
        [-0.6286,  0.5146],
        [-0.1049,  0.3977]], requires_grad=True), Parameter containing:
tensor([[ 0.2273, -0.5302],
        [ 0.1421,  0.1698],
        [-0.4734, -0.3355],
        [ 0.2411,  0.1267], ==> weight_hh_I0
        [-0.3008, -0.2141],
        [ 0.6476, -0.1308],
        [ 0.3987,  0.3062],
        [-0.4571, -0.6013]], requires_grad=True), Parameter containing:
tensor([ 0.6787,  0.0369,  0.4847,  0.1465,  0.2274,  0.5282,  0.6705, -0.4692],
       requires_grad=True), Parameter containing:
tensor([ 0.0884,  0.5277,  0.5123,  0.4393, -0.5117, -0.5092, -0.4276,  0.0888],
       requires_grad=True)]]

<class 'tuple'>: (tensor([[[-0.3256, -0.2448]]]), tensor([[[-0.6155, -0.4567]]]))

tensor([[[-0.1002, -0.0527]],

        [[-0.1197,  0.0175]],

        [[-0.1471,  0.0656]],

        [[-0.3598, -0.1478]],

        [[-0.3256, -0.2448]]])
"""


sentence = ['benim', 'adim', 'ne', '?']
tags = [0,1,0,1]
sentence = [0, 1, 2, 3]
char_sentence = [['b', 'e', 'n', 'i', 'm'], ['a', 'd', 'i', 'm'], ['n', 'e'], ['?']]
char_sentence = [[0, 1, 2, 3, 4], [5, 6, 3, 4, 8], [2, 1,8,8,8], [7,8,8,8,8]]

char_sentence=torch.tensor(char_sentence, dtype=torch.long)
sentence=torch.tensor(sentence, dtype=torch.long)

mdl =  CharEmbLSTMTagger(embedding_dim=6, char_embedding_dim=2, hidden_dim=5, charset_size=9, vocab_size=4, tagset_size=2)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(mdl.parameters(), lr=0.1)

with torch.no_grad():
    # inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = mdl(char_sentence, sentence)
    print(tag_scores)


