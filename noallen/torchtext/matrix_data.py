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

def smoothed_sampling(instances, alpha=None, num_neg_samples=1):
    unique, counts = np.unique(instances, return_counts=True, axis=0)
    unique_idxs = np.arange(0, unique.shape[0])
    if alpha is not None:
        counts = np.power(counts, alpha)
    probs = counts.astype('float') / counts.sum()
    sample_idxs = np.random.choice(unique_idxs, size=instances.shape[0]*num_neg_samples, replace=True, p=probs)
    sample = np.take(unique, sample_idxs, axis=0)
    return sample

def uniform_type_sampling(instances, scores_matrix, indxs_matrix):
    # (num_ins, topk)
    batch_indxs = np.take(indxs_matrix, instances, axis=0)
    # (num_ins, 1))
    sample_idx_idxs = np.random.randint(0, indxs_matrix.shape[1], indxs_matrix.shape[0])
    # import ipdb
    # ipdb.set_trace()
    # (num_ins, 1))
    sample_idxs = np.take(batch_indxs, sample_idx_idxs, axis=1)
    return sample_idxs
def batched_unigram_type_sampling(instances, scores_matrix, indxs_matrix):
    # (num_ins, topk)
    batch_indxs = np.take(indxs_matrix, instances, axis=0)
    # (num_ins, topk)
    batch_scores = np.take(scores_matrix, instances, axis=0)
    # (num_ins, 1))
    sample_idx_idxs = torch.multinomial(torch.from_numpy(batch_scores), 1, replacement=True)
    # import ipdb
    # ipdb.set_trace()
    # (num_ins, 1))
    sample_idxs = np.take(batch_indxs, sample_idx_idxs.cpu().numpy(), axis=1)
    return sample_idxs

def unigram_type_sampling(instances, scores_matrix, indxs_matrix, batch_size=10000):
    samples = []
    for i in range(0, instances.shape[0], batch_size):
        samples.append(batched_unigram_type_sampling(instances[i: i + batch_size], scores_matrix, indxs_matrix))
    return np.concatenate(samples)


def shuffled_sampling(instances):
    return np.random.permutation(instances)

# def sample_compositional(instances, alpha=None, compositional_rels=True, type_scores=None, type_indices=None, num_neg_samples=1):
    # # if alpha is None:
    # np.random.shuffle(instances)
    # subjects, objects, relations = instances[:, 0],  instances[:, 1],  instances[:, 2:]
    # relations = relations if compositional_rels or relations.shape[1] > 1 else relations.reshape(relations.shape[0])
    # sample_fn, kwargs = (smoothed_sampling, {'alpha': alpha}) if alpha is not None else (shuffled_sampling, {})
    # sampled_relations = sample_fn(relations, **kwargs)
    # sample_fn, kwargs = (smoothed_sampling, {'alpha': alpha, 'num_neg_samples': num_neg_samples}) if alpha is not None else (shuffled_sampling, {})
    # sampled_subjects, sampled_objects  = sample_fn(subjects, **kwargs).reshape((instances.shape[0], num_neg_samples)), sample_fn(objects, **kwargs).reshape((instances.shape[0], num_neg_samples))
    # type_sampled_subjects, type_sampled_objects = None, None
    # # if type_scores is not None:
        # # type_sampled_subjects = uniform_type_sampling(subjects, type_scores, type_indices)
        # # type_sampled_objects = uniform_type_sampling(objects, type_scores, type_indices)
    # return  subjects, objects, relations, sampled_relations, sampled_subjects, sampled_objects #, type_sampled_subjects, type_sampled_objects


def sample_compositional(instances, alpha=None, compositional_rels=True, type_scores=None, type_indices=None, num_neg_samples=1):
    # if alpha is None:
    np.random.shuffle(instances)
    sampled_subjects, sampled_objects, subjects, objects, relations = instances[:, 0],  instances[:, 1], instances[:, 2], instances[:, 3], instances[:, 4:]
    sampled_subjects, sampled_objects = sampled_subjects.reshape((instances.shape[0], num_neg_samples)), sampled_objects.reshape((instances.shape[0], num_neg_samples))
    relations = relations if compositional_rels or relations.shape[1] > 1 else relations.reshape(relations.shape[0])
    sample_fn, kwargs = (smoothed_sampling, {'alpha': alpha}) if alpha is not None else (shuffled_sampling, {})
    sampled_relations = sample_fn(relations, **kwargs)
    # sample_fn, kwargs = (smoothed_sampling, {'alpha': alpha, 'num_neg_samples': num_neg_samples}) if alpha is not None else (shuffled_sampling, {})
    # sampled_subjects, sampled_objects  = sample_fn(subjects, **kwargs).reshape((instances.shape[0], num_neg_samples)), sample_fn(objects, **kwargs).reshape((instances.shape[0], num_neg_samples))
    type_sampled_subjects, type_sampled_objects = None, None
    # if type_scores is not None:
        # type_sampled_subjects = uniform_type_sampling(subjects, type_scores, type_indices)
        # type_sampled_objects = uniform_type_sampling(objects, type_scores, type_indices)
    return  subjects, objects, relations, sampled_relations, sampled_subjects, sampled_objects #, type_sampled_subjects, type_sampled_objects

def sample_pairs(instances, alpha=None, compositional_rels=True, type_scores=None, type_indices=None, num_neg_samples=1):
    # if alpha is None:
    np.random.shuffle(instances)
    pairs, relations = instances[:, 0],  instances[:, 1:]
    relations = relations if compositional_rels or relations.shape[1] > 1 else relations.reshape(relations.shape[0])
    sample_fn, kwargs = (smoothed_sampling, {'alpha': alpha, 'num_neg_samples': num_neg_samples}) if alpha is not None else (shuffled_sampling, {})
    sampled_relations = sample_fn(relations, **kwargs)
    # import ipdb
    # ipdb.set_trace()
    sampled_relations = sampled_relations.reshape((relations.shape[0], relations.shape[1], num_neg_samples))
    return pairs, relations, sampled_relations

class TripletIterator():
    def __init__(self, batch_size, fields, pairwise=False, return_nl=False, limit=None, compositional_rels=True, type_scores_file=None, type_indices_file=None, num_neg_samples=1):
        self.batch_size = batch_size
        self.pairwise = pairwise
        self.fields = fields
        self.return_nl = return_nl
        self.limit = limit
        self.compositional_rels = compositional_rels
        self.num_neg_samples = num_neg_samples
        self.type_scores = None if type_scores_file is None else np.load(type_scores_file)
        self.type_indices = None if type_indices_file is None else np.load(type_indices_file)

    def __call__(self, data, device=-1, train=True):
        batches = self._create_batches(data, device, train)
        for batch in batches:
            yield batch


    def _create_batches(self, instance_gen, device=-1, train=True):
        for instances in instance_gen:
            start = 0
            # instances = instances[:500]
            #inputs = subjects, objects, relations, sampled_relations, sampled_subjects, sampled_objects
            sample = sample_pairs if self.pairwise else sample_compositional
            inputs = instances if not train else sample(instances, 0.75, self.compositional_rels, self.type_scores, self.type_indices, self.num_neg_samples)
            # import ipdb
            # ipdb.set_trace()
            for num, batch_start in enumerate(range(0, inputs[0].shape[0], self.batch_size)):
                tensors = tuple(Variable(torch.LongTensor(x[batch_start: batch_start + self.batch_size]), requires_grad=False) for x in inputs)
                if device == None:
                    tensors = tuple([t.cuda() if t is not None else None for t in tensors])
                if self.return_nl:
                    subject_nl = [self.fields[0].vocab.itos[i] for i in inputs[0][batch_start: batch_start + self.batch_size]]
                    object_nl = [self.fields[1].vocab.itos[i] for i in inputs[1][batch_start: batch_start + self.batch_size]]
                    relation_nl = []
                    for rel in  inputs[2][batch_start: batch_start + self.batch_size]:
                        relation_nl += [' '.join([self.fields[2].vocab.itos[j] for j in rel])]
                    yield tensors, (subject_nl, object_nl, relation_nl)
                else:
                    yield tensors

def create_vocab(config, field):
    vocab_path = os.path.join(config.triplet_dir, "vocab.txt")
    tokens = None
    with open(vocab_path) as f:
        text = f.read()
        tokens = text.rstrip().split('\n')
    # specials = list(OrderedDict.fromkeys(tok for tok in [field.unk_token, field.pad_token, field.init_token, field.eos_token] if tok is not None))
    specials = ['<unk>', '<pad>', '<X>', '<Y>'] if config.compositional_rels else ['<unk>', '']
    #vocab = Vocab(tokens, specials=specials, vectors='glove.6B.300d', vectors_cache='/glove')
    vocab = Vocab(tokens, specials=specials, vectors='fasttext.en.300d', vectors_cache='data/fasttext')
    #vocab = Vocab(tokens, specials=specials)
    field.vocab  = vocab

def read(filenames):
    for fname in filenames:
        if os.path.isfile(fname):
            instances = np.load(fname)
            logger.info('Loading {} instances from {}'.format(instances.shape[0], fname))
            yield instances

def read_dev(fname, pairwise=False, limit=None, compositional_rels=True, type_scores_file=None, type_indices_file=None, num_neg_samples=1):
    instances = np.load(fname)
    instances = instances[:limit] if limit is not None else instances
    logger.info('Loading {} instances from {}'.format(instances.shape[0], fname))
    type_scores = None if type_scores_file is None else np.load(type_scores_file)
    type_indices = None if type_indices_file is None else np.load(type_indices_file)
    sample = sample_pairs if pairwise else sample_compositional
    return sample(instances, alpha=.75, compositional_rels=compositional_rels, type_scores=type_scores, type_indices=type_indices, num_neg_samples=num_neg_samples)

def dev_data(sample):
    yield sample

def create_dataset(config, triplet_dir=None):
    triplet_dir = config.triplet_dir if triplet_dir is None else triplet_dir
    #files = [os.path.join(config.triplet_dir, fname) for fname in os.listdir(config.triplet_dir) if fname.endswith('.npy')]
    files = [os.path.join(triplet_dir, 'triplets_' + str(i) + '.npy') for i in range(1, 1000)]
    train_data = _LazyInstances(lambda : iter(read(files[1:])))
    type_scores_file = config.type_scores_file if hasattr(config, 'type_scores_file') else None
    type_indices_file = config.type_indices_file if hasattr(config, 'type_indices_file') else None
    validation_sample = read_dev(files[0], config.pairwise, 500000, config.compositional_rels, type_scores_file, type_indices_file, config.num_neg_samples)
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
    type_scores_file = config.type_scores_file if hasattr(config, 'type_scores_file') else None
    type_indices_file = config.type_indices_file if hasattr(config, 'type_indices_file') else None

    train_iterator = TripletIterator(config.train_batch_size, fields , return_nl=return_nl, 
            compositional_rels=config.compositional_rels, type_scores_file=type_scores_file, type_indices_file=type_indices_file, pairwise=config.pairwise, num_neg_samples=config.num_neg_samples)
    dev_iterator = TripletIterator(config.dev_batch_size, fields, return_nl=return_nl, compositional_rels=config.compositional_rels, pairwise=config.pairwise, num_neg_samples=config.num_neg_samples)

    return train, dev, train_iterator, dev_iterator, args, rels
