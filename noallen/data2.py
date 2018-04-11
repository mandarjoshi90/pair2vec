from allennlp.data import DatasetReader, Vocabulary, Instance, Token
from allennlp.data.fields import TextField, LabelField, MetadataField
from noallen.iterator import BasicSamplingIterator
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.common import Params
from typing import Optional, Dict, Union, Sequence, Iterable
from tqdm import tqdm
from collections import defaultdict
from noallen import util
import os

# This is bad; this belongs in the Vocabulary class. Bad. Bad. Bad.
from allennlp.data.vocabulary import DEFAULT_NON_PADDED_NAMESPACES

@DatasetReader.register('triple_reader')
class TripleReader(DatasetReader):
    def __init__(self, config):
        super().__init__(lazy=True)
        config.relation_namespace = 'tokens' if config.compositional_rels else "relation_labels"
        config.argument_namespace = 'tokens' if config.compositional_args else "arg_labels"
        self.config = config

        self._token_indexers = {'tokens': SingleIdTokenIndexer()}

    def _get_tokens(self, entity):
        return [Token(token) for token in entity.split(self.config.split_on)]

    def get_field(self, string, is_relation):
        if is_relation:
            return TextField(self._get_tokens(string),
                             self._token_indexers) if self.config.compositional_rels else LabelField(string,
                                                                                                     label_namespace='relation_labels')
        else:
            return TextField(self._get_tokens(string),
                             self._token_indexers) if self.config.compositional_args else LabelField(string,
                                                                                                     label_namespace='arg_labels')
    def _read(self, filename: str):
        with open(filename, encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                parts = line.strip().split('\t')
                parts = [part.strip() for part in parts]
                count = int(parts[self.config.count_idx]) if hasattr(self.config, "count_idx") else 1
                instance = self.text_to_instance(parts[self.config.sub_idx], parts[self.config.obj_idx], parts[self.config.rel_idx], count)
                yield instance

    def text_to_instance(self, subject, obj, relation=None, count=1):
        fields = {}
        fields['subjects'] = self.get_field(subject, False)
        fields['objects'] = self.get_field(obj, False)
        metadata = {"count": count}
        fields['metadata'] = MetadataField(metadata)

        if relation is not None:
            fields['observed_relations'] = self.get_field(relation, True)
        instance = Instance(fields)
        return instance

    @classmethod
    def from_params(cls, params: Params) -> 'TripleReader':
        config_file = params.pop('config_file')
        exp = params.pop('experiment', 'multiplication')
        config = util.get_config(config_file, exp)
        return cls(config=config)


def create_dataset(config):
    dataset_reader = TripleReader(config)
    train_data = dataset_reader.read(config.train_data_path)
    validation_data = dataset_reader.read(config.dev_data_path)
    return train_data, validation_data


def vocab_from_instances(train_instances: Iterable['adi.Instance'],
                    dev_instances: Iterable['adi.Instance'],
                   min_count: Dict[str, int] = None,
                   max_vocab_size: Union[int, Dict[str, int]] = None,
                   non_padded_namespaces: Sequence[str] = DEFAULT_NON_PADDED_NAMESPACES,
                   pretrained_files: Optional[Dict[str, str]] = None,
                   only_include_pretrained_words: bool = False) -> 'Vocabulary':

    namespace_token_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for instance in tqdm(train_instances):
        instance.count_vocab_items(namespace_token_counts)
    for instance in tqdm(dev_instances):
        instance.count_vocab_items(namespace_token_counts)

    return Vocabulary(counter=namespace_token_counts,
                      min_count=min_count,
                      max_vocab_size=max_vocab_size,
                      non_padded_namespaces=non_padded_namespaces,
                      pretrained_files=pretrained_files,
                      only_include_pretrained_words=only_include_pretrained_words)

def create_vocab(config, datasets):
    vocab_path = os.path.join(config.save_path, "vocabulary")
    if os.path.exists(vocab_path):
        vocab = Vocabulary.from_files(vocab_path)
    else:
        max_vocab_size = config.max_vocab_size if hasattr(config, "max_vocab_size") else None
        vocab = vocab_from_instances(datasets[0], datasets[1], max_vocab_size=max_vocab_size)
    vocab.save_to_files(vocab_path)
    return vocab


def get_iterator(config, vocab):
    iterator = BasicSamplingIterator("observed_relations", "sampled_relations",
                                     config.batch_size, max_instances_in_memory=config.chunk_size)
    iterator.index_with(vocab)
    return iterator


def read_data(config):
    # if config.compositional_args:
    #     raise NotImplementedError()
    # if config.compositional_rels:
    #     raise NotImplementedError()

    train, dev = create_dataset(config)
    vocab = create_vocab(config, [train, dev])
    config.n_args = vocab.get_vocab_size(config.argument_namespace)
    config.n_rels = vocab.get_vocab_size(config.relation_namespace)

    iterator = get_iterator(config, vocab)
    # import ipdb
    # ipdb.set_trace()

    return train, dev, iterator
