import random
from tqdm import tqdm
from collections import defaultdict

def extract_sentences(sentence_file, pairs, window=5):
    with open(sentence_file) as f:
        tuples = []
        for line in tqdm(f):
            words = line.strip().split(' ')
            valid_word_pairs = [(i,i+w) for i in range(0, len(words) - window) for w in range(1, window+1)]
            tuples += [(words[i], words[j], ' '.join(words[max(i-2, 0):i] + ['X'] + words[i+1:j] + ['Y'] + words[min(len(words) - 1, j+1): min(len(words)-1, j+3)])) for i,j in valid_word_pairs if ((words[i], words[j]) in pairs)]
            tuples += [(words[j], words[i], ' '.join(words[max(i-2, 0):i] + ['Y'] + words[i+1:j] + ['X'] + words[min(len(words) - 1, j+1): min(len(words)-1, j+3)])) for i,j in valid_word_pairs if ((words[j], words[i]) in pairs)]
            #tuples += [(words[i], words[j], ' '.join(words[i-2: j+3])) for i,j in valid_word_pairs if (words[i], words[j]) in pairs or (words[j], words[i]) in pairs]
            #if len(tuples) > 100:
            #    import ipdb
            #    ipdb.set_trace()
            #if len(tuples) > 10:
            #    break
    return tuples

def get_bless_pairs(filenames):
    pairs = []
    for filename in filenames:
        lines = []
        with open(filename) as f:
            text = f.read()
            lines = text.strip().split('\n')
            triples = [line.strip().split('\t') for line in text.strip().split('\n') ]
            pairs += [(w1, w2) for (w1, w2, _) in triples]
    return pairs


def create_triples_dataset(filenames, sentence_file,  output_files):
    pairs = get_bless_pairs(filenames)
    tuples = extract_sentences( sentence_file, set(pairs))
    random.shuffle(tuples)
    with open(outfile, 'w') as f:
        for tup in tuples:
            f.write('\t'.join(tup) + '\n')


#infiles = ['/home/mandar90/data/lexinf/bless/train.txt', '/home/mandar90/data/lexinf/bless/valid.txt',  '/home/mandar90/data/lexinf/bless/test.txt']
infiles = ['/home/mandar90/data/lexinf/bless/train.txt', '/home/mandar90/data/lexinf/bless/valid.txt',  '/home/mandar90/data/lexinf/bless/test.txt','/home/mandar90/data/lexinf/evaluation/train.txt', '/home/mandar90/data/lexinf/evaluation/valid.txt',  '/home/mandar90/data/lexinf/evaluation/test.txt', '/home/mandar90/data/lexinf/root09/train.txt', '/home/mandar90/data/lexinf/root09/valid.txt',  '/home/mandar90/data/lexinf/root09/test.txt', '/home/mandar90/data/lexinf/khn/train.txt', '/home/mandar90/data/lexinf/khn/valid.txt',  '/home/mandar90/data/lexinf/khn/test.txt']
outfile = '/home/mandar90/data/lexinf/combo-sents/all.txt'
sent_file = '/sdb/data/wikipedia-sentences/raw_sentences.txt'
create_triples_dataset(infiles, sent_file, outfile)




