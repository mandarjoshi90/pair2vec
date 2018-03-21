from collections import defaultdict
from collections import Counter
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_top_k(key_to_value_counts, top_k):
    for key in key_to_value_counts:
        key_to_value_counts[key] = key_to_value_counts[key].most_common(top_k)
    return key_to_value_counts

def create_relation_candidates(triples, top_k=2):
    entity_to_subject_relations, entity_to_object_relations = defaultdict(set), defaultdict(set)
    relations_to_arguments = defaultdict(set)

    logger.info("Relation to arguments")
    for e1, relation, e2 in triples:
        entity_to_subject_relations[e1].add(relation)
        entity_to_object_relations[e2].add(relation)
        relations_to_arguments[relation].add((e1, e2))

    relation_to_subject_type, relation_to_object_type = defaultdict(Counter), defaultdict(Counter)
    logger.info("Relation to entity counts")
    for relation, args in relations_to_arguments.items():
        for e1,e2 in args:
            for subject_relation_type in entity_to_subject_relations[e1]:
                relation_to_subject_type[relation][subject_relation_type] += 1
            for object_relation_type in entity_to_object_relations[e2]:
                relation_to_object_type[relation][object_relation_type] += 1
    # filter
    logger.info("Relation to topk counts")
    relation_to_object_type = get_top_k(relation_to_object_type, top_k)
    relation_to_subject_type = get_top_k(relation_to_subject_type, top_k)

    # construct candidate set
    logger.info("Relation to candidate set")
    relation_to_candidate_subjects, relation_to_candidate_objects = {}, {}
    for relation in relation_to_subject_type:
        for subject_type in relation_to_subject_type[relation]:
            for e1, _ in relations_to_arguments[subject_type]:
                relation_to_candidate_subjects[relation].add(e1)
    for relation in relation_to_object_type:
        for object_type in relation_to_object_type[relation]:
            for _, e2 in relations_to_arguments[object_type]:
                relation_to_candidate_objects[relation].add(e2)
    return relation_to_candidate_subjects, relation_to_candidate_objects


def slurp_file(data_file):
    with open(data_file, encoding='utf-8') as f:
        data = f.read()
    return data

def write_to_file(data_file, dict_object):
    with open(data_file, 'w', encoding='utf-8') as f:
        for key, values in dict_object.items():
            key = [key]  if isinstance(key, str) else list(key)
            for part in key:
                f.write(part)
                f.write('\t')
            for value in values:
                f.write(value)
                f.write('\t')
            f.write('\n')
        f.close()


def to_triples(data):
    lines = data.strip().split('\n')
    triples = set()
    for line in lines:
        parts = line.strip().split('\t')
        parts = [part.strip() for part in parts]
        triples.add((parts[0], parts[1], parts[2]))
    return triples

def build_graph(triples, subject_relation_to_objects, object_relation_to_subjects):
    for e1, relation, e2 in triples:
        subject_relation_to_objects[(e1, relation)].add(e2)
        object_relation_to_subjects[(e2, relation)].add(e1)

def preprocess(data_dir, files=('train.txt', 'valid.txt', 'test.txt'), preprocessed_dir='preprocessed'):
    subject_relation_to_objects, object_relation_to_subjects = defaultdict(set), defaultdict(set)
    relation_to_candidate_subjects, relation_to_candidate_objects = None, None
    logger.info("Start preprocessing")
    for file in files:
        data = slurp_file(os.path.join(data_dir, file))
        triples = to_triples(data)
        logger.info("Constructing true set for {}".format(file))
        build_graph(triples, subject_relation_to_objects, object_relation_to_subjects)

        if file == 'train.txt':
            logger.info("Constructing relation candidate from {}".format(file))
            relation_to_candidate_subjects, relation_to_candidate_objects = create_relation_candidates(triples, 3)
    preprocess_dir = (os.path.join(data_dir, preprocessed_dir))
    if not os.path.exists(preprocess_dir):
        os.makedirs(preprocess_dir)
    # write preprocessed files
    logger.info("Writing to disk")
    write_to_file(os.path.join(preprocess_dir, 'relation_to_candidate_objects.tsv'), relation_to_candidate_objects)
    write_to_file(os.path.join(preprocess_dir, 'relation_to_candidate_subjects.tsv'), relation_to_candidate_subjects)
    write_to_file(os.path.join(preprocess_dir, 'subject_relation_to_objects.tsv'), subject_relation_to_objects)
    write_to_file(os.path.join(preprocess_dir, 'object_relation_to_subjects.tsv'), object_relation_to_subjects)

import sys
preprocess(sys.argv[1])