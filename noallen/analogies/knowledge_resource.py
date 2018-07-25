import bsddb
import random
from tqdm import tqdm
from collections import defaultdict

class KnowledgeResource:
    """
    Holds the resource graph data
    """
    def __init__(self, resource_prefix):
        """
        Init the knowledge resource
        :param resource_prefix - the resource directory and file prefix
        """
        self.term_to_id = bsddb.btopen(resource_prefix + '_term_to_id.db', 'r')
        self.id_to_term = bsddb.btopen(resource_prefix + '_id_to_term.db', 'r')
        self.path_to_id = bsddb.btopen(resource_prefix + '_path_to_id.db', 'r')
        self.id_to_path = bsddb.btopen(resource_prefix + '_id_to_path.db', 'r')
        self.l2r_edges = bsddb.btopen(resource_prefix + '_l2r.db', 'r')

    def get_term_by_id(self, id):
        return self.id_to_term[str(id)]

    def get_path_by_id(self, id):
        return self.id_to_path[str(id)]

    def get_id_by_term(self, term):
        return int(self.term_to_id[term]) if self.term_to_id.has_key(term) else -1

    def get_id_by_path(self, path):
        return int(self.path_to_id[path]) if self.path_to_id.has_key(path) else -1

    def get_relations(self, x, y):
        """
        Returns the relations from x to y
        """
        path_dict = {}
        key = str(x) + '###' + str(y)
        path_str = self.l2r_edges[key] if self.l2r_edges.has_key(key) else ''

        if len(path_str) > 0:
            paths = [tuple(map(int, p.split(':'))) for p in path_str.split(',')]
            path_dict = { path : count for (path, count) in paths }

        return path_dict

def print_paths():
    word1, word2 = 'attendant', 'captain'
    word1_id, word2_id = ks.get_id_by_term(word1), ks.get_id_by_term(word2)
    path_dict = ks.get_relations(word1_id, word2_id)
    print('Found {} paths'.format(len(path_dict)))
    for k, v in path_dict.items():
        print(ks.get_path_by_id(k), v)

def get_path_tokens(path):
    edges = path.split('_')
    tokens = [edge_parts.split('/')[0] for edge_parts in edges]
    return ' '.join(tokens)


def create_triples_dataset(filenames, ks, output_file, min_count=0):
    tuples = []
    for filename in filenames:
        lines = []
        with open(filename) as f:
            text = f.read()
            lines += text.strip().split('\n')
        pairs = [line.strip().split('\t') for line in lines ]
        #import pdb
        #pdb.set_trace()
        for word1, word2, _ in tqdm(pairs):
            word1_id, word2_id = ks.get_id_by_term(word1), ks.get_id_by_term(word2)
            path_dict = ks.get_relations(word1_id, word2_id)
            token_dict = defaultdict(int)
            for path, count in path_dict.items():
                token_dict[get_path_tokens(ks.get_path_by_id(path))] += count
            tuples += [(word1, word2, p, str(c)) for p, c in token_dict.items()]
    random.shuffle(tuples)
    with open(output_file, 'w') as f:
        for tup in tuples:
            if int(tup[-1]) > min_count:
                f.write('\t'.join(tup) + '\n')


ks = KnowledgeResource('/sdb/data/paths-wikipedia/wiki')
infiles = ['/home/mandar90/data/lexinf/bless/train.txt', '/home/mandar90/data/lexinf/bless/valid.txt',  '/home/mandar90/data/lexinf/bless/test.txt']
#outfiles = ['/home/mandar90/data/lexinf/bless-paths/train.txt', '/home/mandar90/data/lexinf/bless-paths/valid.txt']
outfile = '/home/mandar90/data/lexinf/bless-paths/all.txt'
create_triples_dataset(infiles, ks, outfile, 3)




