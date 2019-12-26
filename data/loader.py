"""
Data loader for TACRED json files.
"""

import json
import random
import torch
import numpy as np

from utils import constant
from utils.vocab import Vocab

class DataLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    """
    def __init__(self, filename, batch_size, opt, vocab, evaluation=False):
        self.batch_size = batch_size
        self.opt = opt
        self.vocab = vocab
        self.eval = evaluation
        self.label2id = constant.LABEL_TO_ID


        data = self.preprocess(filename, vocab, opt)

        # shuffle for training
        if not evaluation:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
        self.id2label = dict([(v,k) for k,v in self.label2id.items()])
        self.labels = [self.id2label[d[-1]] for d in data]
        self.num_examples = len(data)

        # chunk into batches
        data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        self.data = data
        print("{} batches created for {}".format(len(data), filename))

    def preprocess(self, filename, vocab, opt):
        """ Preprocess the data and convert to ids. """
        with open(filename, "r", encoding="utf-8") as f:

            processed = []
            for d in f.readlines():
                d = eval(d.replace("\n",""))
                tokens = list(d['sentence'])
                if opt['lower']:
                    tokens = [t.lower() for t in tokens]

                for i in d["M_position"]:
                        tokens[i] = 'SUBJ-'+d['subj_type']
                for i in d["LP_position"]:
                        tokens[i] = 'OBJ-' + d['obj_type']
                ner = ['<PAD>'] * len(tokens)
                depl = []
                for de in d["depl"]:
                       depl = depl + de
                tokens = map_to_ids(tokens, vocab.word2id)
                pos = map_to_ids(d['pos'], constant.POS_TO_ID)
                ner = map_to_ids(ner, constant.NER_TO_ID)
                deprel = map_to_ids(depl, constant.DEPREL_TO_ID)
                head = d["head"]
                l = len(tokens)
                subj_positions = get_positions(d['M_position'], l)
                obj_positions = get_positions(d['LP_position'], l)
                subj_type = [constant.SUBJ_NER_TO_ID[d['subj_type']]]
                obj_type = [constant.OBJ_NER_TO_ID[d['obj_type']]]
                if d['label'] != 'False':
                    d['label'] = 'Lives_In/Exhibits'
                relation = self.label2id[d['label']]

                processed += [(tokens, pos, ner, deprel, head, subj_positions, obj_positions, subj_type, obj_type, relation)]
            return processed

    def gold(self):
        """ Return gold labels as a list. """
        return self.labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch_size = len(batch)
        batch = list(zip(*batch))
        assert len(batch) == 10

        # sort all fields by lens for easy RNN operations
        lens = [len(x) for x in batch[0]]
        batch, orig_idx = sort_all(batch, lens)

        # word dropout
        if not self.eval:
            words = [word_dropout(sent, self.opt['word_dropout']) for sent in batch[0]]
        else:
            words = batch[0]

        # convert to tensors
        words = get_long_tensor(words, batch_size)
        masks = torch.eq(words, 0)
        pos = get_long_tensor(batch[1], batch_size)
        ner = get_long_tensor(batch[2], batch_size)
        deprel = get_long_tensor(batch[3], batch_size)
        head = generate_adj(batch[4])
        subj_positions = get_long_tensor(batch[5], batch_size)
        obj_positions = get_long_tensor(batch[6], batch_size)
        subj_type = get_long_tensor(batch[7], batch_size)
        obj_type = get_long_tensor(batch[8], batch_size)

        rels = torch.LongTensor(batch[9])

        return (words, masks, pos, ner, deprel, head, subj_positions, obj_positions, subj_type, obj_type, rels, orig_idx)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)


def map_to_ids(tokens, vocab):
    ids = [vocab[t] if t in vocab else constant.UNK_ID for t in tokens]
    return ids
def generate_adj(heads_list):

    adjs = []
    head_len = max(len(x) for x in heads_list)
    for head in heads_list:
        adj = torch.FloatTensor(head_len, head_len).fill_(0)
        for i in range(head_len):
            adj[i][i] = 1
        for i, index_list in enumerate(head):
            for s in index_list:
                if s != 0:
                    adj[i][s-1] = 1
                    adj[s-1][i] = 1
        adjs.append(adj)
    adjs = torch.stack(adjs)
    return adjs

def get_positions(positions, length):

    new_position = list(range(-positions[0], 0)) + [300]*(positions[-1] - positions[0] + 1) + \
            list(range(1, length-positions[-1]))
    for i in positions:
            new_position[i] = 0
    return new_position


def get_long_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded LongTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.LongTensor(batch_size, token_len).fill_(constant.PAD_ID)
    for i, s in enumerate(tokens_list):
        tokens[i, :len(s)] = torch.LongTensor(s)
    return tokens


def sort_all(batch, lens):
    """ Sort all fields by descending order of lens, and return the original indices. """
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]


def word_dropout(tokens, dropout):
    """ Randomly dropout tokens (IDs) and replace them with <UNK> tokens. """
    return [constant.UNK_ID if x != constant.UNK_ID and np.random.random() < dropout \
            else x for x in tokens]

if __name__ == "__main__":
    opt = {"data_dir": "D:/Cross-sentence/AGGCN_TACRED-master/dataset/tacred","batch_size": 50,
           "vocab_dir": "D:/Cross-sentence/AGGCN_TACRED-master/dataset/vocab", "word_dropout": 0.04,
           "lower": False}
    vocab_file = opt['vocab_dir'] + '/vocab.pkl'
    vocab = Vocab(vocab_file, load=True)
    train_batch = DataLoader(opt['data_dir'] + '/train.json', opt['batch_size'], opt, vocab, evaluation=False)
    dev_batch = DataLoader(opt['data_dir'] + '/dev.json', opt['batch_size'], opt, vocab, evaluation=True)
