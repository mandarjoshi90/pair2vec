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

import logging
logger = logging.getLogger(__name__)

def smoothed_sampling(instances, alpha=None):
    unique, counts = np.unique(instances, return_counts=True, axis=0)
    unique_idxs = np.arange(0, unique.shape[0])
    if alpha is not None:
        counts = np.power(counts, alpha)
    probs = counts.astype('float') / counts.sum()
    sample_idxs = np.random.choice(unique_idxs, size=instances.shape[0], replace=True, p=probs)
    sample = np.take(unique, sample_idxs, axis=0)
    return sample

def shuffled_sampling(instances):
    return np.random.permutation(instances)

def sample(instances, alpha=None):
    if alpha is None:
        np.random.shuffle(instances)
    subjects, objects, relations = instances[:, 0],  instances[:, 1],  instances[:, 2:]
    sample_fn, kwargs = (smoothed_sampling, {'alpha': alpha}) if alpha is not None else (shuffled_sampling, {})
    sampled_subjects, sampled_objects, sampled_relations = sample_fn(subjects, **kwargs), sample_fn(objects, **kwargs), sample_fn(relations, **kwargs)
    return  subjects, objects, relations, sampled_relations, sampled_subjects, sampled_objects



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
            #if self.limit is not None:
            #    instances = instances[:self.limit]
            start = 0
            #inputs = subjects, objects, relations, sampled_relations, sampled_subjects, sampled_objects
            inputs = instances if not train else sample(instances, 0.75)
            effective_vocab_sizes = [len(f.vocab) - len(f.vocab.specials) for f in self.fields]
            #inputs = tuple((ip%efs) + len(self.fields[i].vocab.specials) for i, (ip, efs) in enumerate(zip(inputs, effective_vocab_sizes)))
            #while True:
            for num, batch_start in enumerate(range(0, inputs[0].shape[0], self.batch_size)):
                tensors = tuple(Variable(torch.LongTensor(x[batch_start: batch_start + self.batch_size]), requires_grad=False) for x in inputs)
                if device == None:
                    tensors = tuple([t.cuda() for t in tensors])
                if self.return_nl:
                    subject_nl = [self.fields[0].vocab.itos[i] for i in inputs[0][batch_start: batch_start + self.batch_size]]
                    object_nl = [self.fields[1].vocab.itos[i] for i in inputs[1][batch_start: batch_start + self.batch_size]]
                    relation_nl = []
                    for rel in  inputs[2][batch_start: batch_start + self.batch_size]:
                        #import ipdb
                        #ipdb.set_trace()
                        relation_nl += [' '.join([self.fields[2].vocab.itos[j] for j in rel])]
                    yield tensors, (subject_nl, object_nl, relation_nl)
                else:
                    yield tensors
                #if num > 5:
                #    break

def create_vocab(config, field):
    vocab_path = os.path.join(config.triplet_dir, "vocab.txt")
    tokens = None
    with open(vocab_path) as f:
        text = f.read()
        tokens = text.rstrip().split('\n')
    # specials = list(OrderedDict.fromkeys(tok for tok in [field.unk_token, field.pad_token, field.init_token, field.eos_token] if tok is not None))
    specials = ['<unk>', '<pad>', '<X>', '<Y>']
    #vocab = Vocab(tokens, specials=specials, vectors='glove.6B.200d', vectors_cache='/glove')
    vocab = Vocab(tokens, specials=specials)
    field.vocab  = vocab

def read(filenames):
    for fname in filenames:
        if os.path.isfile(fname):
            instances = np.load(fname)
            logger.info('Loading {} instances from {}'.format(instances.shape[0], fname))
            yield instances

def read_dev(fname, limit=None):
    instances = np.load(fname)
    instances = instances[:limit] if limit is not None else instances
    logger.info('Loading {} instances from {}'.format(instances.shape[0], fname))
    return sample(instances)

def dev_data(sample):
    yield sample

def create_dataset(config):
    triplet_dir = config.triplet_dir
    #files = [os.path.join(config.triplet_dir, fname) for fname in os.listdir(config.triplet_dir) if fname.endswith('.npy')]
    files = [os.path.join(config.triplet_dir, 'triplets_' + str(i) + '.npy') for i in range(1, 1000)]
    train_data = _LazyInstances(lambda : iter(read(files[4:])))
    validation_sample = read_dev(files[0], 500000)
    validation_data = _LazyInstances(lambda : iter (dev_data(validation_sample)))
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
    dev_iterator = TripletIterator(config.dev_batch_size, fields, return_nl=return_nl)

    return train, dev, train_iterator, dev_iterator, args, rels
