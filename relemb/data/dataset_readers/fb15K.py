from relemb.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.common import Params
from overrides import overrides
from relemb import util
from typing import Dict
from allennlp.common.tqdm import Tqdm
from allennlp.data.fields import Field, ListField, LabelField
from allennlp.data.instance import Instance
import os
import logging

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
@DatasetReader.register("fb15k")
class FB15KReader(DatasetReader):
    def __init__(self, data_dir: str, relation_to_candidate_subjects_file: str,
                 relation_to_candidate_objects_file: str,
                 relation_subjects_to_objects_file: str,
                 relation_objects_to_subjects_file: str,
                 use_relation_type_constraints: bool,
                 preprocessed_dir: str = 'preprocessed') -> None:
        super().__init__(False)
        self._use_relation_type_constraints = use_relation_type_constraints
        self._data_dir = data_dir
        preprocessed_dir = os.path.join(data_dir, preprocessed_dir)
        if use_relation_type_constraints:
            self._relation_subjects = read_key_to_values_file(os.path.join(preprocessed_dir, relation_to_candidate_subjects_file), 1)
            self._relation_objects = read_key_to_values_file(os.path.join(preprocessed_dir, relation_to_candidate_objects_file), 1)
        self._relation_subjects_to_objects = read_key_to_values_file(os.path.join(preprocessed_dir, relation_subjects_to_objects_file), 2)
        self._relation_objects_to_subjects = read_key_to_values_file(os.path.join(preprocessed_dir, relation_objects_to_subjects_file), 2)


    @overrides
    def _read(self, params: Params):
        triple_file = os.path.join(self._data_dir, params.pop('triples_file'))
        instances = []
        with open(triple_file, encoding='utf-8') as f:
            lines = f.readlines()
            for line in Tqdm.tqdm(lines):
                parts = line.strip().split('\t')
                parts = [part.strip() for part in parts]
                instance = self.text_to_instance(parts[0], parts[1], parts[2])
                instances.append(instance)
        logger.info("Creating {} instances".format(len(instances)))
        return instances

    @overrides
    def text_to_instance(self, subject, relation, object):
        fields: Dict[str, Field] = {}
        fields['subjects'] = LabelField(subject, label_namespace='argument_labels')
        fields['objects'] = LabelField(object, label_namespace='argument_labels')
        fields['relations'] = LabelField(relation, label_namespace='relation_labels')
        # import ipdb
        # ipdb.set_trace()
        fields['valid_objects'] = ListField([LabelField(obj, label_namespace='argument_labels') for obj in self._relation_subjects_to_objects[(subject, relation)]])
        fields['valid_subjects'] = ListField(
            [LabelField(arg, label_namespace='argument_labels') for arg in self._relation_objects_to_subjects[(object, relation)]])
        if self._use_relation_type_constraints:
            fields['object_candidates'] = ListField(
                [LabelField(obj, label_namespace='argument_labels') for obj in self._relation_objects.get((relation), [])])
            fields['subject_candidates'] = ListField(
                [LabelField(arg, label_namespace='argument_labels') for arg in self._relation_subjects.get((relation), [])])
        instance = Instance(fields)
        return instance

    @classmethod
    def from_params(cls, params: Params) -> 'FB15KReader':
        data_dir = params.pop('data_dir')
        relation_subjects_file = params.pop('relation_to_candidate_subjects_file', 'relation_to_candidate_subjects.tsv')
        relation_objects_file = params.pop('relation_to_candidate_objects_file', 'relation_to_candidate_objects.tsv')
        relation_subjects_to_objects_file = params.pop('relation_subjects_to_objects_file', 'subject_relation_to_objects.tsv')
        relation_objects_to_subjects_file = params.pop('relation_objects_to_subjects_file', 'object_relation_to_subjects.tsv')
        use_relation_type_constraints = params.pop('relation_objects_to_subjects_file', False)
        return cls(data_dir=data_dir, relation_to_candidate_subjects_file=relation_subjects_file,
                   relation_to_candidate_objects_file=relation_objects_file,
                   relation_subjects_to_objects_file=relation_subjects_to_objects_file,
                   relation_objects_to_subjects_file=relation_objects_to_subjects_file,
                   use_relation_type_constraints=use_relation_type_constraints)


def read_key_to_values_file(filename: str, key_parts: int):
    data = util.slurp_file(filename)
    lines = data.strip().split('\n')
    dict_object = {}
    for line in lines:
        parts = line.strip().split('\t')
        key = tuple(parts[:key_parts])
        dict_object[key] = parts[key_parts: ]
    return dict_object
