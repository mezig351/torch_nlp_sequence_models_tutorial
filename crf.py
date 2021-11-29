# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2017-12-04 23:19:38
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2018-12-16 22:15:56
from __future__ import print_function

import datetime
import itertools
import string

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch import optim
from torch.nn.utils.rnn import pad_sequence

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

t0 = datetime.datetime.now()
# torch.manual_seed(1)

START_TAG = -2
STOP_TAG = -1
EMBEDDING_DIM = 5
HIDDEN_DIM = 4
CHAR_EMBEDDING_DIM = 4
MORPH_EMBEDDING_DIM = 4

# Make up some training data
training_data = [(
    "the wall street journal reported today that apple corporation made money".split(),
    "B I I I O O O B I O O".split(),
    "A3sg|P3sg|Loc A3sg|P3sg|Loc A3sg|P3sg|Loc A3sg|P3sg|Loc A3sg|P3sg|Loc A3sg|P3sg|Loc A3sg|P3sg|Loc A3sg|P3sg|Loc A3sg|P3sg|Loc A3sg|P3sg|Loc A3sg|P3sg|Loc".split()
), (
    "georgia tech is a university in georgia".split(),
    "B I O O O O B".split(),
    "A3sg|P3sg|Loc A3sg|P3sg|Loc A3sg|P3sg|Loc A3sg|P3sg|Loc A3sg|P3sg|Loc A3sg|P3sg|Loc A3sg|P3sg|Loc".split()
)]


word_to_ix = {}
for sentence, _, _ in training_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)+1
word_to_ix['<PAD>'] = 0

tag_to_ix = {"B": 1, "I": 2, "O": 3}

char_to_ix = {char: enum + 1 for (enum, char) in enumerate(string.printable + 'ğüşıöçÜŞİÖÇ')}
char_to_ix['<PAD>'] = 0
char_to_ix['<START>'] = len(char_to_ix)
char_to_ix['<STOP>'] = len(char_to_ix)

morphchar_to_ix = {}
for c in 'A3sg|P3sg|Loc':
    if c not in morphchar_to_ix:
        morphchar_to_ix[c] = len(morphchar_to_ix) + 1
morphchar_to_ix['<PAD>'] = 0
morphchar_to_ix['<START>'] = len(morphchar_to_ix)
morphchar_to_ix['<STOP>'] = len(morphchar_to_ix)

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

def prepare_char_sentence(seq, to_ix):
    seq = [[to_ix[c] for c in w] for w in seq]
    seq[0] = [to_ix['<START>']] + seq[0]
    seq[-1] = seq[-1] + [to_ix['<STOP>']]
    return seq

def prepare_char_sequence(seq):
    seq = [torch.LongTensor(word) for word in seq]
    lengths = torch.LongTensor(list(map(len, seq)))
    seq = pad_sequence(seq, batch_first=True, padding_value=0)
    lengths, perm_idx = lengths.sort(0, descending=True)
    seq = seq[perm_idx]
    # seq = seq.transpose(0, 1)
    # .view(1, len(training_data[0][0]))
    # seq = seq.view(1, seq.shape[0], seq.shape[1])
    # lengths = lengths.view(1, -1)
    # perm_idx = perm_idx.view(1, -1)
    return seq, lengths, perm_idx

# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec, m_size):
    """
    calculate log of exp sum
    args:
        vec (batch_size, vanishing_dim, hidden_dim) : input tensor
        m_size : hidden_dim
    return:
        batch_size, hidden_dim
    """
    _, idx = torch.max(vec, 1)  # B * 1 * M
    max_score = torch.gather(vec, 1, idx.view(-1, 1, m_size)).view(-1, 1, m_size)  # B * M
    return max_score.view(-1, m_size) + torch.log(torch.sum(torch.exp(vec - max_score.expand_as(vec)), 1)).view(-1, m_size)  # B * M

# class myDataParallel(nn.DataParallel):
#     def __init__(self, model):
#         super(myDataParallel, self).__init__(model)
#         self.model = model
#
#     def neg_log_likelihood(self, sentence, char_sentence, char_lengths, perm_idx, tags):
#         return self.model.neg_log_likelihood(sentence, char_sentence, char_lengths, perm_idx, tags)


class CRF(nn.Module):

    def __init__(self, tagset_size, vocab_size, charset_size, morph_charset_size, gpu): # embedding_dim, hidden_dim, char_embedding_dim,
        super(CRF, self).__init__()
        print("build CRF...")
        self.gpu = gpu
        # Matrix of transition parameters.  Entry i,j is the score of transitioning from i to j.
        self.tagset_size = tagset_size
        self.char_embeds = nn.Embedding(charset_size, CHAR_EMBEDDING_DIM, padding_idx=0)
        self.morphchar_embeds = nn.Embedding(morph_charset_size, MORPH_EMBEDDING_DIM, padding_idx=0)
        self.word_embeds = nn.Embedding(vocab_size, EMBEDDING_DIM, padding_idx=0)
        self.lstm_char = nn.LSTM(CHAR_EMBEDDING_DIM, CHAR_EMBEDDING_DIM// 2,
                            num_layers=1, bidirectional=True, batch_first=True)
        self.lstm_morphchar = nn.LSTM(MORPH_EMBEDDING_DIM, MORPH_EMBEDDING_DIM // 2,
                                      num_layers=1, bidirectional=True, batch_first=True)
        self.lstm = nn.LSTM(EMBEDDING_DIM+CHAR_EMBEDDING_DIM+MORPH_EMBEDDING_DIM, HIDDEN_DIM // 2,
                            num_layers=1, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(HIDDEN_DIM, self.tagset_size+2)

        # # We add 2 here, because of START_TAG and STOP_TAG
        # # transitions (f_tag_size, t_tag_size), transition value from f_tag to t_tag
        init_transitions = torch.zeros(self.tagset_size+2, self.tagset_size+2)
        init_transitions[:,START_TAG] = -10000.0
        init_transitions[STOP_TAG,:] = -10000.0
        init_transitions[:,0] = -10000.0
        init_transitions[0,:] = -10000.0
        if torch.cuda.is_available():
            init_transitions = init_transitions.cuda(self.gpu)
        self.transitions = nn.Parameter(init_transitions)

        # self.transitions = nn.Parameter(torch.Tensor(self.tagset_size+2, self.tagset_size+2))
        # self.transitions.data.zero_()

    def _calculate_PZ(self, feats, mask):
        """
            input:
                feats: (batch, seq_len, self.tag_size+2)
                masks: (batch, seq_len)
        """
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(2)
        # print feats.view(seq_len, tag_size)
        assert(tag_size == self.tagset_size+2)
        mask = mask.transpose(1,0).contiguous()
        ins_num = seq_len * batch_size
        ## be careful the view shape, it is .view(ins_num, 1, tag_size) but not .view(ins_num, tag_size, 1)
        feats = feats.transpose(1,0).contiguous().view(ins_num,1, tag_size).expand(ins_num, tag_size, tag_size)
        ## need to consider start
        scores = feats + self.transitions.view(1,tag_size,tag_size).expand(ins_num, tag_size, tag_size)
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)
        # build iter
        seq_iter = enumerate(scores)
        _, inivalues = next(seq_iter)  # bat_size * from_target_size * to_target_size
        # only need start from start_tag
        partition = inivalues[:, START_TAG, :].clone().view(batch_size, tag_size, 1)  # bat_size * to_target_size

        ## add start score (from start to all tag, duplicate to batch_size)
        # partition = partition + self.transitions[START_TAG,:].view(1, tag_size, 1).expand(batch_size, tag_size, 1)
        # iter over last scores
        for idx, cur_values in seq_iter:
            # previous to_target is current from_target
            # partition: previous results log(exp(from_target)), #(batch_size * from_target)
            # cur_values: bat_size * from_target * to_target

            cur_values = cur_values + partition.contiguous().view(batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
            cur_partition = log_sum_exp(cur_values, tag_size)
            # print cur_partition.data

                # (bat_size * from_target * to_target) -> (bat_size * to_target)
            # partition = utils.switch(partition, cur_partition, mask[idx].view(bat_size, 1).expand(bat_size, self.tagset_size)).view(bat_size, -1)
            mask_idx = mask[idx, :].view(batch_size, 1).expand(batch_size, tag_size)

            ## effective updated partition part, only keep the partition value of mask value = 1
            masked_cur_partition = cur_partition.masked_select(mask_idx)
            ## let mask_idx broadcastable, to disable warning
            mask_idx = mask_idx.contiguous().view(batch_size, tag_size, 1)

            ## replace the partition where the maskvalue=1, other partition value keeps the same
            partition.masked_scatter_(mask_idx, masked_cur_partition)
        # until the last state, add transition score for all partition (and do log_sum_exp) then select the value in STOP_TAG
        cur_values = self.transitions.view(1,tag_size, tag_size).expand(batch_size, tag_size, tag_size) + partition.contiguous().view(batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
        cur_partition = log_sum_exp(cur_values, tag_size)
        final_partition = cur_partition[:, STOP_TAG]
        return final_partition.sum(), scores

    def _get_char_hidden(self, char_sentence, char_lengths, perm_idx, char_or_morph):
        if char_or_morph == 'char':
            char_embeds = self.char_embeds(char_sentence)
            packed = pack_padded_sequence(char_embeds, char_lengths, batch_first=True)
            packed_out, (hidden, cell) = self.lstm_char(packed)
        elif char_or_morph == 'morph':
            char_embeds = self.morphchar_embeds(char_sentence)
            packed = pack_padded_sequence(char_embeds, char_lengths, batch_first=True)
            packed_out, (hidden, cell) = self.lstm_morphchar(packed)
        else:
            raise ValueError('char_or_morph variable should be either \'char\' or \'morph\'')
        # sort tuple of original index and permutation index according to permutation index in order to reverse the sort
        reverse_perm = sorted(list(zip(range(len(char_sentence)), perm_idx.numpy())), key=lambda x: x[1])
        # get rid of permutation index to use it as index slicer; to get the right hidden layer of character embeddings
        # to add to word embeddings
        reverse_perm = [x[0] for x in reverse_perm]
        # reshape from 3d to 2d to match shape of word embeddings
        hidden = hidden.view(len(char_sentence), -1)
        hidden = hidden[reverse_perm]
        return hidden

    def _get_lstm_features(self, sentence_batch, char_sentence, char_lengths, perm_idx, batch_morph_input, batch_morph_lengths, batch_morph_perm):
        batch_size = sentence_batch.shape[0]
        batch_length = sentence_batch.shape[1]

        hidden = self._get_char_hidden(char_sentence, char_lengths, perm_idx, 'char')
        hidden = hidden.view(batch_size, batch_length, CHAR_EMBEDDING_DIM)

        hidden_morph = self._get_char_hidden(batch_morph_input, batch_morph_lengths, batch_morph_perm, 'morph')
        hidden_morph = hidden_morph.view(batch_size, batch_length, MORPH_EMBEDDING_DIM)

        word_embeds = self.word_embeds(sentence_batch)
        word_char_embeds = torch.cat((word_embeds, hidden, hidden_morph), dim=2)

        # embeds = self.word_embeds(sentence_batch).view(batch_size, batch_length, -1)
        embeds = word_char_embeds.view(batch_size, batch_length, -1)
        lstm_out, _ = self.lstm(embeds)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats


    def _viterbi_decode(self, feats, mask):
        """
            input:
                feats: (batch, seq_len, self.tag_size+2)
                mask: (batch, seq_len)
            output:
                decode_idx: (batch, seq_len) decoded sequence
                path_score: (batch, 1) corresponding score for each sequence (to be implementated)
        """
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(2)
        assert(tag_size == self.tagset_size+2)
        ## calculate sentence length for each sentence
        length_mask = torch.sum(mask.long(), dim = 1).view(batch_size,1).long()
        ## mask to (seq_len, batch_size)
        mask = mask.transpose(1,0).contiguous()
        ins_num = seq_len * batch_size
        ## be careful the view shape, it is .view(ins_num, 1, tag_size) but not .view(ins_num, tag_size, 1)
        feats = feats.transpose(1,0).contiguous().view(ins_num, 1, tag_size).expand(ins_num, tag_size, tag_size)
        ## need to consider start
        scores = feats + self.transitions.view(1,tag_size,tag_size).expand(ins_num, tag_size, tag_size)
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)

        # build iter
        seq_iter = enumerate(scores)
        ## record the position of best score
        back_points = list()
        partition_history = list()
        ##  reverse mask (bug for mask = 1- mask, use this as alternative choice)
        # mask = 1 + (-1)*mask
        mask =  (1 - mask.long()).byte()
        _, inivalues = next(seq_iter)  # bat_size * from_target_size * to_target_size
        # only need start from start_tag
        partition = inivalues[:, START_TAG, :].clone().view(batch_size, tag_size)  # bat_size * to_target_size
        # print "init part:",partition.size()
        partition_history.append(partition)
        # iter over last scores
        for idx, cur_values in seq_iter:
            # previous to_target is current from_target
            # partition: previous results log(exp(from_target)), #(batch_size * from_target)
            # cur_values: batch_size * from_target * to_target
            cur_values = cur_values + partition.contiguous().view(batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
            ## forscores, cur_bp = torch.max(cur_values[:,:-2,:], 1) # do not consider START_TAG/STOP_TAG
            # print "cur value:", cur_values.size()
            partition, cur_bp = torch.max(cur_values, 1)
            # print "partsize:",partition.size()
            # exit(0)
            # print partition
            # print cur_bp
            # print "one best, ",idx
            partition_history.append(partition)
            ## cur_bp: (batch_size, tag_size) max source score position in current tag
            ## set padded label as 0, which will be filtered in post processing
            cur_bp.masked_fill_(mask[idx].view(batch_size, 1).expand(batch_size, tag_size), 0)
            back_points.append(cur_bp)
        # exit(0)
        ### add score to final STOP_TAG
        partition_history = torch.cat(partition_history, 0).view(seq_len, batch_size, -1).transpose(1,0).contiguous() ## (batch_size, seq_len. tag_size)
        ### get the last position for each setences, and select the last partitions using gather()
        last_position = length_mask.view(batch_size,1,1).expand(batch_size, 1, tag_size) -1
        last_partition = torch.gather(partition_history, 1, last_position).view(batch_size,tag_size,1)
        ### calculate the score from last partition to end state (and then select the STOP_TAG from it)
        last_values = last_partition.expand(batch_size, tag_size, tag_size) + self.transitions.view(1,tag_size, tag_size).expand(batch_size, tag_size, tag_size)
        _, last_bp = torch.max(last_values, 1)
        pad_zero = autograd.Variable(torch.zeros(batch_size, tag_size)).long()
        if torch.cuda.is_available():
            pad_zero = pad_zero.cuda(self.gpu)
        back_points.append(pad_zero)
        back_points  =  torch.cat(back_points).view(seq_len, batch_size, tag_size)

        ## select end ids in STOP_TAG
        pointer = last_bp[:, STOP_TAG]
        insert_last = pointer.contiguous().view(batch_size,1,1).expand(batch_size,1, tag_size)
        back_points = back_points.transpose(1,0).contiguous()
        ## move the end ids(expand to tag_size) to the corresponding position of back_points to replace the 0 values
        # print "lp:",last_position
        # print "il:",insert_last
        back_points.scatter_(1, last_position, insert_last)
        # print "bp:",back_points
        # exit(0)
        back_points = back_points.transpose(1,0).contiguous()
        ## decode from the end, padded position ids are 0, which will be filtered if following evaluation
        decode_idx = autograd.Variable(torch.LongTensor(seq_len, batch_size))
        if torch.cuda.is_available():
            decode_idx = decode_idx.cuda(self.gpu)
        decode_idx[-1] = pointer.data # detach()
        for idx in range(len(back_points)-2, -1, -1):
            pointer = torch.gather(back_points[idx], 1, pointer.contiguous().view(batch_size, 1))
            decode_idx[idx] = pointer.data.t() # feili  pointer.detach().view(batch_size)
        path_score = None
        decode_idx = decode_idx.transpose(1,0)
        return path_score, decode_idx

    def forward(self, batch_sentences, mask, char_sentence, char_lengths, perm_idx, batch_morph_input, batch_morph_lengths, batch_morph_perm):
        feats = self._get_lstm_features(batch_sentences, char_sentence, char_lengths, perm_idx, batch_morph_input, batch_morph_lengths, batch_morph_perm)
        path_score, best_path = self._viterbi_decode(feats, mask)
        return path_score, best_path
        

    def _score_sentence(self, scores, mask, tags):
        """
            input:
                scores: variable (seq_len, batch, tag_size, tag_size)
                mask: (batch, seq_len)
                tags: tensor  (batch, seq_len)
            output:
                score: sum of score for gold sequences within whole batch
        """
        # Gives the score of a provided tag sequence
        batch_size = scores.size(1)
        seq_len = scores.size(0)
        tag_size = scores.size(2)
        ## convert tag value into a new format, recorded label bigram information to index  
        new_tags = autograd.Variable(torch.LongTensor(batch_size, seq_len))
        if torch.cuda.is_available():
            new_tags = new_tags.cuda(self.gpu)
        for idx in range(seq_len):
            if idx == 0:
                ## start -> first score
                new_tags[:,0] =  (tag_size - 2)*tag_size + tags[:,0]

            else:
                new_tags[:,idx] =  tags[:,idx-1]*tag_size + tags[:,idx]

        ## transition for label to STOP_TAG
        end_transition = self.transitions[:,STOP_TAG].contiguous().view(1, tag_size).expand(batch_size, tag_size)
        ## length for batch,  last word position = length - 1
        length_mask = torch.sum(mask.long(), dim = 1).view(batch_size,1).long()
        ## index the label id of last word
        end_ids = torch.gather(tags, 1, length_mask - 1)

        ## index the transition score for end_id to STOP_TAG
        end_energy = torch.gather(end_transition, 1, end_ids)

        ## convert tag as (seq_len, batch_size, 1)
        new_tags = new_tags.transpose(1,0).contiguous().view(seq_len, batch_size, 1)
        ### need convert tags id to search from 400 positions of scores
        tg_energy = torch.gather(scores.view(seq_len, batch_size, -1), 2, new_tags).view(seq_len, batch_size)  # seq_len * bat_size
        ## mask transpose to (seq_len, batch_size)
        tg_energy = tg_energy.masked_select(mask.transpose(1,0))
        
        # ## calculate the score from START_TAG to first label
        # start_transition = self.transitions[START_TAG,:].view(1, tag_size).expand(batch_size, tag_size)
        # start_energy = torch.gather(start_transition, 1, tags[0,:])

        ## add all score together
        # gold_score = start_energy.sum() + tg_energy.sum() + end_energy.sum()
        gold_score = tg_energy.sum() + end_energy.sum()
        return gold_score

    def neg_log_likelihood(self, sents, mask, tags, char_sentence, char_lengths, perm_idx, batch_morph_input, batch_morph_lengths, batch_morph_perm):
        # nonegative log likelihood
        feats = self._get_lstm_features(sents, char_sentence, char_lengths, perm_idx, batch_morph_input, batch_morph_lengths, batch_morph_perm)
        forward_score, scores = self._calculate_PZ(feats, mask)
        gold_score = self._score_sentence(scores, mask, tags)
        # print "batch, f:", forward_score.data[0], " g:", gold_score.data[0], " dis:", forward_score.data[0] - gold_score.data[0]
        # exit(0)
        # if self.average_batch:
        #     return (forward_score - gold_score) / batch_size
        # else:
        return forward_score - gold_score


model = CRF(tagset_size=len(tag_to_ix), vocab_size=len(word_to_ix), charset_size=len(char_to_ix), morph_charset_size=len(morphchar_to_ix), gpu=False)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

# Check predictions before training
with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix).view(1, len(training_data[0][0]))
    mask = torch.ones(precheck_sent.shape)
    char_inputs, char_lengths, perm_idx = prepare_char_sequence(prepare_char_sentence(training_data[0][0], char_to_ix))
    morph_inputs, moprh_lengths, morph_perm = prepare_char_sequence(prepare_char_sentence(training_data[0][2], morphchar_to_ix))
    print(model(precheck_sent, mask, char_inputs, char_lengths, perm_idx, morph_inputs, moprh_lengths, morph_perm))

X_tr = []
y_tr = []
char_tr = []
morph_tr = []
for sentence, tags, morphs in training_data*100:
    X_tr.append(prepare_sequence(sentence, word_to_ix))
    y_tr.append(prepare_sequence(tags, tag_to_ix))
    char_tr.append(prepare_char_sentence(sentence, char_to_ix))
    morph_tr.append(prepare_char_sentence(morphs, morphchar_to_ix))


batch_size = 32
# Make sure prepare_sequence from earlier in the LSTM section is loaded
for epoch in range(10):  # again, normally you would NOT do 300 epochs, it is toy data
    for index in range(0, len(X_tr), batch_size):
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is,
        # turn them into Tensors of word indices.
        batch_sentences = X_tr[index:index + batch_size]
        batch_sentences = pad_sequence(batch_sentences, batch_first=True, padding_value=0)
        mask = batch_sentences != 0
        batch_tags = y_tr[index:index + batch_size]
        batch_tags = pad_sequence(batch_tags, batch_first=True, padding_value=0)

        batch_char_input = char_tr[index:index + batch_size]
        sentence_lengths = list(map(len, batch_char_input))
        pads = [[[char_to_ix['<PAD>']]]*(batch_sentences.shape[1]-n) for n in sentence_lengths]
        batch_char_input = [a+b for (a,b) in zip(batch_char_input, pads)]
        batch_char_input = list(itertools.chain.from_iterable(batch_char_input))
        batch_char_input, batch_char_lengths, batch_perm_idcs = prepare_char_sequence(batch_char_input)

        batch_morph_input = morph_tr[index:index + batch_size]
        sentence_lengths = list(map(len, batch_morph_input))
        pads = [[[morphchar_to_ix['<PAD>']]] * (batch_sentences.shape[1] - n) for n in sentence_lengths]
        batch_morph_input = [a + b for (a, b) in zip(batch_morph_input, pads)]
        batch_morph_input = list(itertools.chain.from_iterable(batch_morph_input))
        batch_morph_input, batch_morph_lengths, batch_morph_perm = prepare_char_sequence(batch_morph_input)

        # Step 3. Run our forward pass.
        loss = model.neg_log_likelihood(batch_sentences, mask, batch_tags, batch_char_input, batch_char_lengths,
                                        batch_perm_idcs, batch_morph_input, batch_morph_lengths, batch_morph_perm)
        # print(loss)
        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss.backward()
        optimizer.step()
            # i+=1

# Check predictions after training
with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix).view(1, len(training_data[0][0]))
    mask = torch.ones(precheck_sent.shape)
    char_inputs, char_lengths, perm_idx = prepare_char_sequence(prepare_char_sentence(training_data[0][0], char_to_ix))
    morph_inputs, moprh_lengths, morph_perm = prepare_char_sequence(
        prepare_char_sentence(training_data[0][2], morphchar_to_ix))
    print(model(precheck_sent, mask, char_inputs, char_lengths, perm_idx, morph_inputs, moprh_lengths, morph_perm))


t1 = datetime.datetime.now()
print('Total Time:', t1-t0)










