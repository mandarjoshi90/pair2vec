from embeddings.vocab import Vocab
from embeddings.indexed_field import Field
from torch.autograd import Variable
import torch
import numpy as np

from typing import Optional, Dict, Union, Sequence, Iterable, Iterator, TypeVar, List
from tqdm import tqdm
from collections import defaultdict, Counter, OrderedDict
from embeddings import util
import os
import random

import logging
logger = logging.getLogger(__name__)

# From AllenNLP
class _LazyInstances(Iterable):
    """
    An ``Iterable`` that just wraps a thunk for generating instances and calls it for
    each call to ``__iter__``.
    """
    def __init__(self, instance_generator) -> None:
        super().__init__()
        self.instance_generator = instance_generator

    def __iter__(self):
        instances = self.instance_generator()
        return instances


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

def sample_compositional(instances, alpha=None, compositional_rels=True, type_scores=None, type_indices=None, num_neg_samples=1, num_sampled_relations=1, model_type='sampling'):
    np.random.shuffle(instances)
    subjects, objects, relations = instances[:, 0],  instances[:, 1],  instances[:, 2:]
    relations = relations if compositional_rels or relations.shape[1] > 1 else relations.reshape(relations.shape[0])
    sample_fn, kwargs = (smoothed_sampling, {'alpha': alpha, 'num_neg_samples': num_sampled_relations}) if alpha is not None else (shuffled_sampling, {})
    sampled_relations = sample_fn(relations, **kwargs)
    sampled_relations = sampled_relations.reshape((relations.shape[0], relations.shape[1], num_sampled_relations))
    sample_fn, kwargs = (smoothed_sampling, {'alpha': alpha, 'num_neg_samples': num_neg_samples}) if alpha is not None else (shuffled_sampling, {})
    sampled_subjects, sampled_objects  = sample_fn(subjects, **kwargs).reshape((instances.shape[0], num_neg_samples)), sample_fn(objects, **kwargs).reshape((instances.shape[0], num_neg_samples))
    return  subjects, objects, relations, sampled_relations, sampled_subjects, sampled_objects #, type_sampled_subjects, type_sampled_objects



class TripletIterator():
    def __init__(self, batch_size, fields, return_nl=False, limit=None, compositional_rels=True, type_scores_file=None, type_indices_file=None, num_neg_samples=1,
            alpha=0.75, num_sampled_relations=1, model_type='sampling'):
        self.batch_size = batch_size
        self.fields = fields
        self.return_nl = return_nl
        self.limit = limit
        self.alpha = alpha
        self.compositional_rels = compositional_rels
        self.num_neg_samples = num_neg_samples
        self.num_sampled_relations = num_sampled_relations
        self.model_type = model_type
        self.type_scores = None if type_scores_file is None else np.load(type_scores_file)
        self.type_indices = None if type_indices_file is None else np.load(type_indices_file)

    def __call__(self, data, device=-1, train=True):
        batches = self._create_batches(data, device, train)
        for batch in batches:
            yield batch


    def _create_batches(self, instance_gen, device=-1, train=True):
        for instances in instance_gen:
            start = 0
            sample = sample_compositional
            inputs = instances if (not train) else sample(instances, self.alpha, self.compositional_rels, self.type_scores, self.type_indices, self.num_neg_samples, self.num_sampled_relations, model_type=self.model_type)
            for num, batch_start in enumerate(range(0, inputs[0].shape[0], self.batch_size)):
                tensors = tuple(Variable(torch.LongTensor(x[batch_start: batch_start + self.batch_size]), requires_grad=False) for x in inputs)
                if device == None:
                    tensors = tuple([t.cuda() if t is not None else None for t in tensors])
                if self.return_nl:
                    relation_nl = []
                    rel_index = 2
                    for rel in  inputs[rel_index][batch_start: batch_start + self.batch_size]:
                        relation_nl += [' '.join([self.fields[rel_index].vocab.itos[j] for j in rel])]
                    yield tensors, (relation_nl)
                else:
                    yield tensors

def create_vocab(config, field):
    vocab_path = getattr(config, 'vocab_file', os.path.join(config.triplet_dir, "vocab.txt"))
    tokens = None
    with open(vocab_path) as f:
        text = f.read()
        tokens = text.rstrip().split('\n')
    specials = ['<unk>', '<pad>', '<X>', '<Y>'] if config.compositional_rels else ['<unk>', '']
    init_with_pretrained = getattr(config, 'init_with_pretrained', True)
    vectors, vectors_cache = (None, None) if not init_with_pretrained else (getattr(config, 'word_vecs', 'fasttext.en.300d'), getattr(config, 'word_vecs_cache', 'data/fasttext'))
    vocab = Vocab(tokens, specials=specials, vectors=vectors, vectors_cache=vectors_cache)
    field.vocab  = vocab

def read(filenames):
    for fname in filenames:
        if os.path.isfile(fname):
            instances = np.load(fname)
            logger.info('Loading {} instances from {}'.format(instances.shape[0], fname))
            yield instances

def read_dev(fname, limit=None, compositional_rels=True, type_scores_file=None, type_indices_file=None, num_neg_samples=1, num_sampled_relations=1, model_type='sampling'):
    instances = np.load(fname)
    instances = instances[:limit] if limit is not None else instances
    logger.info('Loading {} instances from {}'.format(instances.shape[0], fname))
    type_scores = None if type_scores_file is None else np.load(type_scores_file)
    type_indices = None if type_indices_file is None else np.load(type_indices_file)
    sample = sample_compositional
    return sample(instances, alpha=.75, compositional_rels=compositional_rels, type_scores=type_scores, type_indices=type_indices, num_neg_samples=num_neg_samples, num_sampled_relations=num_sampled_relations, model_type=model_type)

def dev_data(sample):
    yield sample

def create_dataset(config, triplet_dir=None):
    triplet_dir = config.triplet_dir if triplet_dir is None else triplet_dir
    #files = [os.path.join(config.triplet_dir, fname) for fname in os.listdir(config.triplet_dir) if fname.endswith('.npy')]
    files = [os.path.join(triplet_dir, 'triplets_' + str(i) + '.npy') for i in range(1, 1000)]
    train_data = _LazyInstances(lambda : iter(read(files[1:])))
    type_scores_file = config.type_scores_file if hasattr(config, 'type_scores_file') else None
    type_indices_file = config.type_indices_file if hasattr(config, 'type_indices_file') else None
    model_type = getattr(config, 'model_type', 'sampling')
    validation_sample = read_dev(files[0], 500000, config.compositional_rels, type_scores_file, type_indices_file, config.num_neg_samples, config.num_sampled_relations, model_type)
    validation_data = _LazyInstances(lambda : iter (dev_data(validation_sample)))
    return train_data, validation_data

def read_data(config, return_nl=False, preindex=True):
    args = Field(lower=True, batch_first=True) 
    rels = Field(lower=True, batch_first=True) if config.compositional_rels else Field(batch_first=True)
    fields = [args, args, rels]
    train, dev = create_dataset(config)
    create_vocab(config, args)
    rels.vocab = args.vocab
    config.n_args = len(args.vocab)
    config.n_rels = len(rels.vocab)
    sample_arguments = getattr(config, "sample_arguments", True)
    fields = fields + [rels, args, args] if sample_arguments else fields  + [rels]
    type_scores_file = config.type_scores_file if hasattr(config, 'type_scores_file') else None
    type_indices_file = config.type_indices_file if hasattr(config, 'type_indices_file') else None
    model_type = getattr(config, 'model_type', 'sampling')

    train_iterator = TripletIterator(config.train_batch_size, fields , return_nl=return_nl,
            compositional_rels=config.compositional_rels, type_scores_file=type_scores_file, type_indices_file=type_indices_file, num_neg_samples=config.num_neg_samples,
            alpha=getattr(config, 'alpha', 0.75), num_sampled_relations=getattr(config, 'num_sampled_relations', 1), model_type=model_type)
    dev_iterator = TripletIterator(config.dev_batch_size, fields, return_nl=return_nl, compositional_rels=config.compositional_rels, num_neg_samples=config.num_neg_samples,
            alpha=getattr(config, 'alpha', 0.75), num_sampled_relations=getattr(config, 'num_sampled_relations', 1), model_type=model_type)

    return train, dev, train_iterator, dev_iterator, args, rels
