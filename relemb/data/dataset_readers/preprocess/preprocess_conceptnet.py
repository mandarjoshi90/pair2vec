import os
import gzip
import sys
from tqdm import tqdm
from collections import defaultdict
from sklearn.model_selection import train_test_split

def valid_entity(entity, allow_multiple=False):
    eng = entity.startswith('/c/en/')
    return eng and (allow_multiple or '_' not in entity)

def canonicalize(entity):
    parts = entity.split('/')
    if len(parts) == 5 or len(parts) == 4:
        return parts[3]
    else:
        import ipdb
        ipdb.set_trace()


def get_canonical_tuples(tuples):
    canon_tuples = set()
    for tupl in tuples:
        canon_tuples.add((canonicalize(tupl[0]), tupl[1], canonicalize(tupl[2])))
    return canon_tuples

def get_args_to_relations(tuples):
    args_to_rels = defaultdict(set)
    for tupl in tuples:
        args_to_rels[(tupl[0], tupl[2])].add(tupl[1])
    return args_to_rels


def split_dataset(tuples, out_dir, split=(0.95, 0.05)):
    tuples = list(get_canonical_tuples(tuples))
    args_to_rels = get_args_to_relations(tuples)
    split_data = train_test_split(tuples, train_size=split[0], test_size=split[1])
    args_to_rels_train = get_args_to_relations(split_data[0])
    args_to_rels_valid = get_args_to_relations(split_data[1])
    print('Writing train: {} examples'.format(len(split_data[0])))
    write_to_file(split_data[0], os.path.join(out_dir, 'train.txt'))

    print('Writing valid: {} examples'.format(len(split_data[1])))
    write_to_file(split_data[1], os.path.join(out_dir, 'valid.txt'))

    print('Writing args to relations file')
    write_to_file([[word1, word2] + list(rels) for (word1, word2), rels in args_to_rels.items()], os.path.join(out_dir, 'arguments_to_relations.txt'))
    write_to_file([[word1, word2] + list(rels) for (word1, word2), rels in args_to_rels_train.items()],
                  os.path.join(out_dir, 'arguments_to_relations_train.txt'))
    write_to_file([[word1, word2] + list(rels) for (word1, word2), rels in args_to_rels_valid.items()],
                  os.path.join(out_dir, 'arguments_to_relations_valid.txt'))

def write_to_file(tuples, filename):
    with open(filename, encoding='utf-8', mode='w') as f:
        for tupl in tuples:
            f.write('\t'.join(tupl) + '\n')

def preprocess(in_filename, out_dir, allow_multiword):
    filtered = []
    in_lines = 0
    entities, relations = set(), set()
    with gzip.open(in_filename, 'rt') as inf:
        for line in tqdm(inf):
            parts = line.split('\t')
            in_lines += 1
            if valid_entity(parts[2], allow_multiword) and valid_entity(parts[3], allow_multiword   ):
                filtered.append([parts[2], parts[1], parts[3], parts[4]])
                entities.add(parts[2])
                entities.add(parts[3])
                relations.add(parts[1])

    print('Writing {} lines from {}'.format(len(filtered), in_lines))
    print('Entities: {}, Relations:{}'.format(len(entities), len(relations)))
    print('Relations:', relations)
    with gzip.open(os.path.join(out_dir, 'filtered.txt.gz'), 'wt') as outf:
        for line in filtered:
            outf.write('\t'.join(line) )
    split_dataset(filtered, out_dir)

if __name__ == '__main__':
    allow_multiword = False
    if len(sys.argv) > 3:
        allow_multiword = bool(sys.argv[3])
    preprocess(sys.argv[1], sys.argv[2], allow_multiword)
