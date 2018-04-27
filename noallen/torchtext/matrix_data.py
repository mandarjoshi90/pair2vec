from noallen.torchtext.vocab import Vocab
from noallen.torchtext.data import _LazyInstances
from noallen.torchtext.indexed_field import Field
from torch.autograd import Variable
import torch
import numpy as np

from typing import Optional, Dict, Union, Sequence, Iterable, Iterator, TypeVar, List
from tqdm import tqdm
from collections import defaultdict, Counter, OrderedDict
from noallen import util
import os
import random

class TripletIterator():
    def __init__(self, batch_size, fields, return_nl=False, limit=None):
        self.batch_size = batch_size
        self.fields = fields
        self.return_nl = return_nl
        self.limit = limit

    def __call__(self, data, device=-1, train=True):
        batches = self._create_batches(data, device, train)
        for batch in batches:
            yield batch


    def _create_batches(self, instance_gen, device=-1, train=True):
        for instances in instance_gen:
            if self.limit is not None:
                instances = instances[:self.limit]
            np.random.shuffle(instances)
            subjects, objects, relations = instances[:, 0],  instances[:, 1],  instances[:, 2]
            sampled_subjects, sampled_objects, sampled_relations = subjects.copy(), objects.copy(), relations.copy()
            np.random.shuffle(sampled_subjects)
            np.random.shuffle(sampled_objects)
            np.random.shuffle(sampled_relations)

            start = 0
            inputs = subjects, objects, relations, sampled_relations, sampled_subjects, sampled_objects
            effective_vocab_sizes = [len(f.vocab) - len(f.vocab.specials) for f in self.fields]
            inputs = tuple((ip%efs) + len(self.fields[i].vocab.specials) for i, (ip, efs) in enumerate(zip(inputs, effective_vocab_sizes)))
            #while True:
            for num, batch_start in enumerate(range(0, instances.shape[0], self.batch_size)):
                tensors = (Variable(torch.LongTensor(x[batch_start: batch_start + self.batch_size]), requires_grad=False) for x in inputs)
                if device == None:
                    tensors = tuple([t.cuda() for t in tensors])
                yield tensors
                #if num > 5:
                #    break

def create_vocab(config, field):
    vocab_path = os.path.join(config.triplet_dir, "vocab.txt")
    tokens = None
    with open(vocab_path) as f:
        text = f.read()
        tokens = text.rstrip().split('\n')
    specials = list(OrderedDict.fromkeys(tok for tok in [field.unk_token, field.pad_token, field.init_token, field.eos_token] if tok is not None))
    vocab = Vocab(tokens, specials=specials, vectors='glove.6B.200d', vectors_cache='/glove')
    #vocab = Vocab(tokens, specials=specials)
    field.vocab  = vocab

def read(filenames):
    for fname in filenames:
        instances = np.load(fname)
        yield instances


def create_dataset(config):
    triplet_dir = config.triplet_dir
    files = [os.path.join(config.triplet_dir, fname) for fname in os.listdir(config.triplet_dir) if fname.endswith('.npy')]
    train_data = _LazyInstances(lambda : iter(read(files[1:])))
    validation_data = _LazyInstances(lambda : iter (read(files[:1])))
    return train_data, validation_data

def read_data(config, return_nl=False, preindex=True):
    args = Field(lower=True, batch_first=True) if config.compositional_args else Field(batch_first=True)
    rels = Field(lower=True, batch_first=True) if config.compositional_rels else Field(batch_first=True)
    fields = [args, args, rels]
    train, dev = create_dataset(config)
    create_vocab(config, args)
    rels.vocab = args.vocab
    config.n_args = len(args.vocab)
    config.n_rels = len(rels.vocab)
    sample_arguments = hasattr(config, "sample_arguments") and config.sample_arguments
    fields = fields + [rels, args, args] if sample_arguments else fields  + [rels]

    train_iterator = TripletIterator(config.train_batch_size, fields , return_nl=return_nl)
    dev_iterator = TripletIterator(config.dev_batch_size, fields, return_nl=return_nl, limit=5000000)

    return train, dev, train_iterator, dev_iterator, args, rels
