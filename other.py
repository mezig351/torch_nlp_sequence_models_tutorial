# -*- coding:utf8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(72)


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return Variable(tensor)


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


class LSTMTagger(nn.Module):
    def __init__(self, word_emb_dim, char_emb_dim, hidden_dim, vocab_size, tagset_size, char_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.char_emb_dim = char_emb_dim

        self.word_embedding = nn.Embedding(vocab_size, word_emb_dim)
        self.char_embedding = nn.Embedding(char_size, char_emb_dim)
        self.char_lstm = nn.LSTM(char_emb_dim, char_emb_dim)
        self.lstm = nn.LSTM(word_emb_dim + char_emb_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence_word, sentence_char, MAX_WORD_LEN):
        # char emb
        sentence_size = sentence_word.size()[0]
        char_emb = self.char_embedding(sentence_char)  # [sentence_size * MAX_WORD_LEN, char_emb_dim]
        try:
            char_emb = char_emb.view(len(sentence_word), MAX_WORD_LEN, -1).permute(1, 0,
                                                                                   2)  # [MAX_WORD_LEN, sentence_size, char_emb_dim]
        except:
            print("char_emb.size():", char_emb.size())

        self.hidden_char = self.initHidden_char(sentence_size)
        char_lstm_out, self.hidden = self.char_lstm(char_emb, self.hidden_char)
        char_embeded = char_lstm_out[-1, :, :].view(sentence_size, -1)

        # word emb
        word_embeded = self.word_embedding(sentence_word)

        embeded = torch.cat((word_embeded, char_embeded), dim=1)
        # print('embeded size:\n', embeded.size())
        self.hidden = self.initHidden()
        lstm_out, self.hidden = self.lstm(embeded.view(sentence_size, 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(sentence_size, -1))
        tag_scores = F.log_softmax(tag_space)
        return tag_scores

    def initHidden(self):
        result = (Variable(torch.zeros(1, 1, self.hidden_dim)),
                  Variable(torch.zeros(1, 1, self.hidden_dim)))
        return result

    def initHidden_char(self, sentence_size):
        result = (Variable(torch.zeros(1, sentence_size, self.char_emb_dim)),
                  Variable(torch.zeros(1, sentence_size, self.char_emb_dim)))
        return result


# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
WORD_EMB_DIM = 6
CHAR_EMB_DIM = 3
HIDDEN_DIM = 6
MAX_WORD_LEN = 8

model = LSTMTagger(WORD_EMB_DIM, CHAR_EMB_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix), len(char_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# before training
print('before training')
sentence_word = prepare_sequence(training_data[1][0], word_to_ix)
sent_chars = []
for w in training_data[1][0]:
    sps = ' ' * (MAX_WORD_LEN - len(w))
    sent_chars.extend(list(sps + w) if len(w) < MAX_WORD_LEN else list(w[:MAX_WORD_LEN]))
sentence_char = prepare_sequence(sent_chars, char_to_ix)

tag_scores = model(sentence_word, sentence_char, MAX_WORD_LEN)
targets = prepare_sequence(training_data[1][1], tag_to_ix)
print(tag_scores)
print('targets:\n', targets)

for epoch in range(300):
    for sentence, tags in training_data:
        model.zero_grad()
        model.hidden = model.initHidden()
        sentence_word = prepare_sequence(sentence, word_to_ix)
        sent_chars = []
        for w in sentence:
            sps = ' ' * (MAX_WORD_LEN - len(w))
            sent_chars.extend(list(sps + w) if len(w) < MAX_WORD_LEN else list(w[:MAX_WORD_LEN]))
        sentence_char = prepare_sequence(sent_chars, char_to_ix)
        # sentence_char = prepare_char(sentence, char_to_ix, max_length=7)

        targets = prepare_sequence(tags, tag_to_ix)
        tag_scores = model(sentence_word, sentence_char, MAX_WORD_LEN)
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()

# after training
print('after training')
sentence_word = prepare_sequence(training_data[0][0], word_to_ix)
sent_chars = []
for w in training_data[0][0]:
    sps = ' ' * (MAX_WORD_LEN - len(w))
    sent_chars.extend(list(sps + w) if len(w) < MAX_WORD_LEN else list(w[:MAX_WORD_LEN]))
sentence_char = prepare_sequence(sent_chars, char_to_ix)

tag_scores = model(sentence_word, sentence_char, MAX_WORD_LEN)
targets = prepare_sequence(training_data[0][1], tag_to_ix)
print(tag_scores)
print('targets:\n', targets)

