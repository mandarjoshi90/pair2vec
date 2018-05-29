from noallen.data import TextField
from typing import Optional, Dict, Union, Sequence, Iterable, Iterator, TypeVar, List
from tqdm import tqdm
from collections import defaultdict, Counter, OrderedDict
from noallen import util
import os
import torch
import itertools
from itertools import islice
from torchtext.vocab import Vocab
from noallen.torchtext.indexed_field import Field
import random

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


A = TypeVar('A')
def lazy_groups_of(iterator: Iterator[A], group_size: int) -> Iterator[List[A]]:
    """
    Takes an iterator and batches the invididual instances into lists of the
    specified size. The last list may be smaller if there are instances left over.
    """
    return iter(lambda: list(islice(iterator, 0, group_size)), [])

def expand_instance_list(instance_list, fields=None):
    if fields is None:
        new_list = list(itertools.chain(*[[(sub, obj, obs_rel) for _ in range(count)] for sub, obj, obs_rel, count in instance_list]))
    else:
        subject_field, object_field, relation_field = fields[:3]
        sub_list, obj_list, rel_list, count_list = zip(*instance_list)
        sub_list, obj_list, rel_list = subject_field.index(list(sub_list)), object_field.index(list(obj_list)), relation_field.index(list(rel_list))
        new_list = list(itertools.chain(*[[(sub, obj, obs_rel) for _ in range(int(count))] for sub, obj, obs_rel, count in zip(sub_list, obj_list, rel_list, count_list)]))
    random.shuffle(new_list)
    return new_list

def add_negative_samples(instance_list, sample_arguments):
    relations = list([r for s, o, r in instance_list])
    random.shuffle(relations)
    if sample_arguments:
        subjects, objects = list([s for s, o, r in instance_list]), list([o for s, o, r in instance_list])
        random.shuffle(subjects)
        random.shuffle(objects)
        instances = [(sub, obj, obs_rel, samp_rel, samp_sub, samp_obj) for (sub, obj, obs_rel), samp_rel, samp_sub, samp_obj in zip(instance_list, relations, subjects, objects)]
    else:
        instances = [(sub, obj, obs_rel, samp_rel) for (sub, obj, obs_rel), samp_rel in zip(instance_list, relations)]
    return instances

class BasicSamplingIterator():
    def __init__(self, batch_size, chunk_size, fields, return_nl=False, preindex=True, sample_arguments=False):
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.fields = fields
        self.return_nl = return_nl
        self.preindex = preindex
        self.sample_arguments = sample_arguments
    def __call__(self, data, device=-1, train=True):
        batches = self._create_batches(data, device, train)
        for batch in batches:
            yield batch

    def _memory_sized_lists(self, instances):
        yield from lazy_groups_of(iter(instances), self.chunk_size)

    def _create_batches(self, instances, device=-1, train=True):
        for instance_list in self._memory_sized_lists(instances):
            # add negative sampling
            instance_list = add_negative_samples(expand_instance_list(instance_list, fields=self.fields if self.preindex else None), self.sample_arguments)
            for batch_instances in lazy_groups_of(iter(instance_list), self.batch_size):
                inputs = list(zip(*batch_instances))
                tensors = self.to_tensors(inputs, device, train)
                if self.return_nl:
                    yield tensors, inputs
                else:
                    yield tensors

    def to_tensors(self, inputs, device=-1, train=True):
        assert len(inputs) == len(self.fields)
        tensors = [self.fields[i].process(inputs[i], device=device, train=train, indexed=(self.preindex)) for i in range(len(inputs))]
        return tuple(tensors)



def text_to_instance(subject, obj, relation, fields, count=1, max_seq_len=10):
    subject_field, object_field, relation_field = fields
    sub = subject_field.preprocess(subject)[:max_seq_len]
    obj = object_field.preprocess(obj)[:max_seq_len]
    observed_rels = relation_field.preprocess(relation)
    return (sub, obj, observed_rels, count)

def read(filename, fields, config):
    has_count = hasattr(config, 'count_idx')
    with open(filename, encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            parts = line.strip().split('\t')
            parts = [part.strip() for part in parts]
            count = int(parts[config.count_idx]) if has_count else 1
            instance = text_to_instance(parts[config.sub_idx], parts[config.obj_idx], parts[config.rel_idx], fields, count, config.max_seq_len)
            if any([len(f) > config.max_seq_len for f in instance[:-1]]):
                continue
            #if line_idx >= 10:
            #    break
            yield instance

def create_dataset(config, fields):
    train_data = _LazyInstances(lambda : iter(read(config.train_data_path, fields, config)))
    validation_data = _LazyInstances(lambda : iter (read(config.dev_data_path, fields, config)))
    return train_data, validation_data



def vocab_from_instances(train_instances,
                    dev_instances,
                    common_vocab=True,
                   ):

    arg_counter = Counter()
    rel_counter = arg_counter if common_vocab else Counter()
    for sub, obj, rel, _ in tqdm(train_instances):
        arg_counter.update(sub + obj)
        rel_counter.update(rel)
    for sub, obj, rel, _ in tqdm(dev_instances):
        arg_counter.update(sub + obj)
        rel_counter.update(rel)
    return arg_counter, rel_counter


def create_vocab(config, datasets, fields):
    vocab_path = os.path.join(config.save_path, "vocabulary.pth")
    common_vocab = config.compositional_args  and config.compositional_rels
    if os.path.exists(vocab_path):
        arg_counter, rel_counter = torch.load(vocab_path)
    else:
        arg_counter, rel_counter = vocab_from_instances(datasets[0], datasets[1])
        torch.save((arg_counter, rel_counter), vocab_path)
    subject_field, object_field, relation_field = fields
    arg_specials = list(OrderedDict.fromkeys(tok for tok in [subject_field.unk_token, subject_field.pad_token, subject_field.init_token, subject_field.eos_token] if tok is not None))
    rel_specials = list(OrderedDict.fromkeys(tok for tok in [relation_field.unk_token, relation_field.pad_token, relation_field.init_token, relation_field.eos_token] if tok is not None))
    max_vocab_size = config.max_vocab_size if hasattr(config, "max_vocab_size") else None
    arg_vocab = Vocab(arg_counter, specials=arg_specials, vectors='glove.6B.200d', vectors_cache='/glove', max_size=max_vocab_size)
    rel_vocab = Vocab(rel_counter, specials=rel_specials, vectors='glove.6B.200d', vectors_cache='/glove', max_size=max_vocab_size) if not common_vocab else arg_vocab
    subject_field.vocab, object_field.vocab, relation_field.vocab = arg_vocab, arg_vocab, rel_vocab


def read_data(config, return_nl=False, preindex=True):
    args = Field(lower=True, batch_first=True) if config.compositional_args else Field(batch_first=True)
    rels = Field(lower=True, batch_first=True) if config.compositional_rels else Field(batch_first=True)
    fields = [args, args, rels]
    train, dev = create_dataset(config, fields)
    create_vocab(config, [train, dev], fields)
    config.n_args = len(args.vocab)
    config.n_rels = len(rels.vocab)
    sample_arguments = hasattr(config, "sample_arguments") and config.sample_arguments
    fields = fields + [rels, args, args] if sample_arguments else fields  + [rels]

    train_iterator = BasicSamplingIterator(config.train_batch_size, config.chunk_size, fields , return_nl=return_nl, preindex=preindex,
            sample_arguments=sample_arguments)
    dev_iterator = BasicSamplingIterator(config.dev_batch_size, config.chunk_size, fields, return_nl=return_nl, preindex=preindex,
            sample_arguments=sample_arguments)

    return train, dev, train_iterator, dev_iterator, args, rels
