from relemb.data.dataset_readers.dataset_reader import RelembDatasetReader
from allennlp.common import Params
from overrides import overrides
from relemb import util
from typing import Dict
from allennlp.common.tqdm import Tqdm
from allennlp.data.fields import Field, ListField, LabelField
from relemb.data.independent_sequence_label_field import IndependentSequenceLabelField
from allennlp.data.instance import Instance
import os
import logging

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
@RelembDatasetReader.register("fb15k")
class FB15KReader(RelembDatasetReader):
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
        self._preprocessed_dir = preprocessed_dir
        if use_relation_type_constraints:
            self._relation_subjects = read_key_to_values_file(os.path.join(preprocessed_dir, relation_to_candidate_subjects_file), 1)
            self._relation_objects = read_key_to_values_file(os.path.join(preprocessed_dir, relation_to_candidate_objects_file), 1)
        self._relation_subjects_to_objects = read_key_to_values_file(os.path.join(preprocessed_dir, relation_subjects_to_objects_file), 2)
        self._relation_objects_to_subjects = read_key_to_values_file(os.path.join(preprocessed_dir, relation_objects_to_subjects_file), 2)


    @overrides
    def _read(self, params: Params):
        triple_file = os.path.join(self._data_dir, params.pop('triples_file'))
        valid_objects_pf = read_key_to_values_file(os.path.join(self._preprocessed_dir, params.pop('valid_objects_pf')), 2)
        valid_subjects_pf = read_key_to_values_file(os.path.join(self._preprocessed_dir, params.pop('valid_subjects_pf')), 2)
        instances = []
        with open(triple_file, encoding='utf-8') as f:
            lines = f.readlines()
            for line in Tqdm.tqdm(lines):
                parts = line.strip().split('\t')
                parts = [part.strip() for part in parts]
                forward_instance = self.text_to_instance(parts[0], parts[1], parts[2], self._relation_subjects_to_objects, valid_objects_pf)
                backward_instance = self.text_to_instance(parts[2], parts[1], parts[0], self._relation_objects_to_subjects, valid_subjects_pf)
                instances.append(forward_instance)
                instances.append(backward_instance)
        logger.info("Creating {} instances".format(len(instances)))
        return instances

    @overrides
    def text_to_instance(self, subject, relation, obj, valid_objects, valid_objects_pf=None):
        fields: Dict[str, Field] = {}
        fields['subjects'] = LabelField(subject, label_namespace='argument_labels')
        fields['objects'] = LabelField(obj, label_namespace='argument_labels')
        fields['relations'] = LabelField(relation, label_namespace='relation_labels')

        if valid_objects_pf is not None:
            fields['valid_objects_pf'] = IndependentSequenceLabelField(
                valid_objects_pf[' '.join([subject, relation])], label_namespace='argument_labels')

        fields['valid_objects'] = IndependentSequenceLabelField(valid_objects[' '.join([subject, relation])],label_namespace='argument_labels')
        instance = Instance(fields)
        return instance

    @classmethod
    def from_params(cls, params: Params) -> 'FB15KReader':
        data_dir = params.pop('data_dir')
        relation_subjects_file = params.pop('relation_to_candidate_subjects_file', 'relation_to_candidate_subjects.tsv')
        relation_objects_file = params.pop('relation_to_candidate_objects_file', 'relation_to_candidate_objects.tsv')
        relation_subjects_to_objects_file = params.pop('relation_subjects_to_objects_file', 'subject_relation_to_objects.tsv')
        relation_objects_to_subjects_file = params.pop('relation_objects_to_subjects_file', 'object_relation_to_subjects.tsv')
        use_relation_type_constraints = params.pop('use_relation_type_constraints', False)
        return cls(data_dir=data_dir, relation_to_candidate_subjects_file=relation_subjects_file,
                   relation_to_candidate_objects_file=relation_objects_file,
                   relation_subjects_to_objects_file=relation_subjects_to_objects_file,
                   relation_objects_to_subjects_file=relation_objects_to_subjects_file,
                   use_relation_type_constraints=use_relation_type_constraints)


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