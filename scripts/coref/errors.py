import json
import ipdb

def slurp_file(fname):
    with open(fname) as f:
        content = f.read()
        return content

def get_cluster_tuples(clusters):
    cluster_set = set()
    for cluster in clusters:
        tuples = tuple([(span[0], span[1]) for span in cluster])
        cluster_set.add(tuples)
    return cluster_set

def print_clusters(clusters, doc):
    for cluster in clusters:
        strings = set()
        for begin, end in cluster:
            strings.add(' '.join([word.lower() for word in doc[begin: end+1]]))
        if len(strings) > 1:
            print(strings)

def analyse(pred_file):
    contents = slurp_file(pred_file)
    lines = contents.strip().split('\n')
    for i, line in enumerate(lines):
        datum = json.loads(line.strip())
        sentences = datum['sentences']
        doc = [token for sentence in sentences for token in sentence]
        gold_cluster_set = get_cluster_tuples(datum['clusters'])
        pred_cluster_set = get_cluster_tuples(datum['predicted_clusters'])
        print_clusters(gold_cluster_set - pred_cluster_set, doc)
        # ipdb.set_trace()
        if i > 100:
            break

analyse('/home/mandar90/data/coref/viz/coref_elmo_full')