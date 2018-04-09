from allennlp.data import DatasetReader, Vocabulary, Instance, Token
from allennlp.data.fields import TextField, LabelField
from noallen.iterator import BasicSamplingIterator
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.common import Params
import os


class TripleReader(DatasetReader):
    def __init__(self, config):
        super().__init__(lazy=True)
        config.relation_namespace = 'tokens' if config.compositional_rels else "relation_labels"
        config.argument_namespace = 'tokens' if config.compositional_args else "arg_labels"
        self.config = config

        self._token_indexers = {'tokens': SingleIdTokenIndexer()}

    def _get_tokens(self, entity):
        return [Token(token) for token in entity.split('_')]

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
            for line in f:
                parts = line.strip().split('\t')
                parts = [part.strip() for part in parts]
                instance = self.text_to_instance(parts[0], parts[2], parts[1])
                yield instance

    def text_to_instance(self, subject, obj, relation=None):
        fields = {}
        fields['subjects'] = self.get_field(subject, False)
        fields['objects'] = self.get_field(obj, False)

        if relation is not None:
            fields['observed_relations'] = self.get_field(relation, True)
        instance = Instance(fields)
        return instance


def create_dataset(config):
    dataset_reader = TripleReader(config)
    train_data = dataset_reader.read(config.train_data_path)
    validation_data = dataset_reader.read(config.dev_data_path)
    return train_data, validation_data


def create_vocab(config, datasets):
    vocab_path = os.path.join(config.save_path, "vocabulary")
    if os.path.exists(vocab_path):
        vocab = Vocabulary.from_files(vocab_path)
    else:
        all_instances = [instance for dataset in datasets for instance in dataset]
        vocab = Vocabulary.from_params(Params({}), all_instances)
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
