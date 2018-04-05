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
from collections import defaultdict

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
@RelembDatasetReader.register("fb15k_pairs")
class FB15KPairReader(RelembDatasetReader):
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

    def _read_train(self, valid_objects_pf):
        instances = []
        for e1_relation, e2_set in Tqdm.tqdm(valid_objects_pf.items()):
            parts = e1_relation.split(' ')
            e1, relation = parts[0], parts[1]
            fields: Dict[str, Field] = {}
            fields['subjects'] = LabelField(e1, label_namespace='argument_labels')
            fields['relations'] = LabelField(relation, label_namespace='relation_labels')
            fields['train_true_objects'] = IndependentSequenceLabelField(list(e2_set), label_namespace='argument_labels')
            instance = Instance(fields)
            instances.append(instance)
        return instances

    def _create_test_instance(self, e1, relation, e2, all_true_e2s, candidate_objects):
        fields: Dict[str, Field] = {}
        fields['subjects'] = LabelField(e1, label_namespace='argument_labels')
        fields['relations'] = LabelField(relation, label_namespace='relation_labels')
        fields['objects'] = LabelField(e2, label_namespace='argument_labels')
        fields['all_true_objects'] = IndependentSequenceLabelField(all_true_e2s, label_namespace='argument_labels')
        instance = Instance(fields)
        return instance

    def _read_test(self, triple_file):
        instances = []
        with open(triple_file, encoding='utf-8') as f:
            lines = f.readlines()
            for line in Tqdm.tqdm(lines):
                parts = line.strip().split('\t')
                parts = [part.strip() for part in parts]
                forward_instance = self._create_test_instance(parts[0], parts[1], parts[2], list(self._relation_subjects_to_objects[' '.join([parts[0], parts[1]])]))
                backward_instance = self._create_test_instance(parts[2], parts[1], parts[0], list(
                    self._relation_objects_to_subjects[' '.join([parts[2], parts[1]])]))
                instances.append(forward_instance)
                instances.append(backward_instance)
        return instances

    @overrides
    def _read(self, params: Params):
        triple_file = os.path.join(self._data_dir, params.pop('triples_file'))
        partition = params.pop('partition')
        valid_objects_pf = read_key_to_values_file(os.path.join(self._preprocessed_dir, params.pop('valid_objects_pf')), 2)
        valid_objects_pf = read_key_to_values_file(os.path.join(self._preprocessed_dir, params.pop('valid_subjects_pf')), 2,valid_objects_pf)
        if partition == 'train':
            instances = self._read_train(valid_objects_pf)
        else:
            instances = self._read_test(triple_file)
        logger.info("Creating {} instances".format(len(instances)))
        return instances

    @classmethod
    def from_params(cls, params: Params) -> 'FB15KPairReader':
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


def read_key_to_values_file(filename: str, key_parts: int, dict_object: defaultdict(set) = None):
    if filename is None:
        return None
    data = util.slurp_file(filename)
    lines = data.strip().split('\n')
    dict_object = defaultdict(set) if dict_object is None else dict_object
    for line in lines:
        parts = line.strip().split('\t')
        key = ' '.join(parts[:key_parts]) if key_parts > 1 else parts[key_parts - 1]
        for part in parts[key_parts: ]:
            dict_object[key].add(part)
    return dict_object