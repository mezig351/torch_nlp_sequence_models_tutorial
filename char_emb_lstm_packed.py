# -*- coding:utf8 -*-
import datetime
import string

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

torch.manual_seed(1)

training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]
word_to_ix = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

char_to_ix = {}
char_to_ix[' '] = len(char_to_ix)
for sent, _ in training_data:
    for word in sent:
        for char in word:
            if char not in char_to_ix:
                char_to_ix[char] = len(char_to_ix)

# print(char_to_ix)
# print('len(char_to_ix):',len(char_to_ix))
# print(word_to_ix)
tag_to_ix = {"DET": 0, "NN": 1, "V": 2}


class CharEmbLSTMTagger(nn.Module):

    def __init__(self, embedding_dim, char_embedding_dim, hidden_dim, charset_size, vocab_size, tagset_size):
        super(CharEmbLSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.char_embeddings = nn.Embedding(charset_size, char_embedding_dim)
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm_char = nn.LSTM(char_embedding_dim, char_embedding_dim, batch_first=True)
        self.lstm_word = nn.LSTM(embedding_dim + char_embedding_dim, hidden_dim, batch_first=True)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, char_sentence, char_lengths, perm_idx, sentence):
        char_embeds = self.char_embeddings(char_sentence)
        packed = pack_padded_sequence(char_embeds, char_lengths, batch_first=True)
        packed_out, (hidden, cell) = self.lstm_char(packed)

        # sort tuple of original index and permutation index according to permutation index in order to reverse the sort
        reverse_perm = sorted(list(zip(range(len(sentence)), perm_idx.numpy())), key=lambda x: x[1])
        # get rid of permutation index to use it as index slicer; to get the right hidden layer of character embeddings
        # to add to word embeddings
        reverse_perm = [x[0] for x in reverse_perm]
        # reshape from 3d to 2d to match shape of word embeddings
        hidden = hidden.view(len(sentence), -1)
        hidden = hidden[reverse_perm]

        word_embeds = self.word_embeddings(sentence)
        word_char_embeds = torch.cat((word_embeds, hidden), dim=1)

        lstm_out, _ = self.lstm_word(word_char_embeds.view(len(sentence), 1, -1))

        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
EMBEDDING_DIM = 6
CHAR_EMBEDDING_DIM = 3
HIDDEN_DIM = 6

char_to_ix = {char: enum + 1 for (enum, char) in enumerate(string.printable + 'ğüşıöçÜŞİÖÇ')}
char_to_ix['<PAD>'] = 0
print(char_to_ix['\''])


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


def prepare_char_sequence(seq):
    seq = [torch.LongTensor([char_to_ix[c] for c in w]) for w in seq]
    lengths = torch.LongTensor(list(map(len, seq)))
    seq = pad_sequence(seq, batch_first=True, padding_value=0)
    lengths, perm_idx = lengths.sort(0, descending=True)
    seq = seq[perm_idx]
    # seq = seq.transpose(0, 1)
    return seq, lengths, perm_idx


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

t0 = datetime.datetime.now()
model = CharEmbLSTMTagger(EMBEDDING_DIM, CHAR_EMBEDDING_DIM, HIDDEN_DIM, len(char_to_ix), len(word_to_ix),
                          len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
# Here we don't need to train, so the code is wrapped in torch.no_grad()
with torch.no_grad():
    char_inputs, char_lengths, perm_idx = prepare_char_sequence(training_data[0][0])
    sentence_in = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(char_inputs, char_lengths, perm_idx, sentence_in)
    print(tag_scores)

for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
    for sentence, tags in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices. char_lengths is required for pack_padded_sequence and perm_idx is required to
        # inverse the sorting from longest to shortest word to original positions in the sentence
        char_sentence_in, char_lengths, perm_idx = prepare_char_sequence(sentence)
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)

        # Step 3. Run our forward pass.
        tag_scores = model(char_sentence_in, char_lengths, perm_idx, sentence_in)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()

# See what the scores are after training
with torch.no_grad():
    char_inputs, char_lengths, perm_idx = prepare_char_sequence(training_data[0][0])
    sentence_in = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(char_inputs, char_lengths, perm_idx, sentence_in)

    # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
    # for word i. The predicted tag is the maximum scoring tag.
    # Here, we can see the predicted sequence below is 0 1 2 0 1
    # since 0 is index of the maximum value of row 1,
    # 1 is the index of maximum value of row 2, etc.
    # Which is DET NOUN VERB DET NOUN, the correct sequence!
    print(tag_scores)
t1 = datetime.datetime.now()
print('Total Time:', t1 - t0)
