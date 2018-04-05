from relemb.data.dataset_readers.dataset_reader import RelembDatasetReader
from allennlp.common import Params
from overrides import overrides
from relemb import util
from typing import Dict
from allennlp.common.tqdm import Tqdm
from allennlp.data.fields import Field, LabelField
from relemb.data.independent_sequence_label_field import IndependentSequenceLabelField
from allennlp.data.instance import Instance
import os
import logging

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@RelembDatasetReader.register("conceptnet")
class ConceptNetReader(RelembDatasetReader):
    def __init__(self, data_dir: str,
                 arguments_to_relations_file: str,
                 preprocessed_dir: str = '') -> None:
        super().__init__(False)
        self._data_dir = data_dir
        preprocessed_dir = os.path.join(data_dir, preprocessed_dir)
        self._preprocessed_dir = preprocessed_dir

        self._arguments_to_relations = read_key_to_values_file(os.path.join(preprocessed_dir, arguments_to_relations_file), 2)


    @overrides
    def _read(self, params: Params):
        triple_file = os.path.join(self._data_dir, params.pop('triples_file'))
        partition_true_relations = read_key_to_values_file(os.path.join(self._preprocessed_dir, params.pop('partition_true_relations')), 2)
        instances = []
        with open(triple_file, encoding='utf-8') as f:
            lines = f.readlines()
            for line in Tqdm.tqdm(lines):
                parts = line.strip().split('\t')
                parts = [part.strip() for part in parts]
                instance = self.text_to_instance(parts[0], parts[1], parts[2], self._arguments_to_relations, partition_true_relations)
                instances.append(instance)
        logger.info("Creating {} instances".format(len(instances)))
        return instances

    @overrides
    def text_to_instance(self, subject, relation, obj, all_true_relations, partition_true_relations=None):
        fields: Dict[str, Field] = {}
        fields['subjects'] = LabelField(subject, label_namespace='argument_labels')
        fields['objects'] = LabelField(obj, label_namespace='argument_labels')
        fields['relations'] = LabelField(relation, label_namespace='relation_labels')

        if partition_true_relations is not None:
            fields['partition_true_relations'] = IndependentSequenceLabelField(
                partition_true_relations[' '.join([subject, obj])], label_namespace='relation_labels')

        fields['all_true_relations'] = IndependentSequenceLabelField(all_true_relations[' '.join([subject, obj])],label_namespace='relation_labels')
        instance = Instance(fields)
        return instance

    @classmethod
    def from_params(cls, params: Params) -> 'ConceptNetReader':
        data_dir = params.pop('data_dir')
        preprocessed_dir = params.pop('preprocessed_dir')
        arguments_to_relations_file = params.pop('arguments_to_relations_file', 'arguments_to_relations.txt')
        return cls(data_dir=data_dir, preprocessed_dir=preprocessed_dir,
                   arguments_to_relations_file=arguments_to_relations_file)


def read_key_to_values_file(filename: str, key_parts: int):
    if filename is None:
        return None
    data = util.slurp_file(filename)
    lines = data.strip().split('\n')
    dict_object = {}
    for line in lines:
        parts = line.strip().split('\t')
        key = ' '.join(parts[:key_parts]) if key_parts > 1 else parts[key_parts - 1]
        dict_object[key] = parts[key_parts: ]
    return dict_object