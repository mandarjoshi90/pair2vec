from collections import Counter
from math import log
from random import Random

from docopt import docopt

from vocab import save_vocabulary, save_count_vocabulary


def main():
    args = docopt("""
    Usage:
        corpus2pair_vocab.py [options] <corpus> <vocab>
    
    Options:
        --thr NUM    The minimal pair count for being in the vocabulary [default: 3]
        --pmi NUM    The minimal PMI for being in the vocabulary [default: 0.0]
    """)
    
    corpus_file = args['<corpus>']
    vocab_file = args['<vocab>']
    thr = int(args['--thr'])
    pmi_thr = float(args['--pmi'])
    
    vocab = Counter()
    with open(corpus_file) as f:
        for iline, line in enumerate(f):
            tokens = list(line.strip().split())
            legit_tokens = [(i, w) for i, w in enumerate(tokens) if w != '-1']
            pairs = [(x, y) for j, (ix, x) in enumerate(legit_tokens[:-1]) for iy, y in legit_tokens[j+1:] if iy > ix+1 and x != y]
            pairs = ['_'.join((x, y)) for x, y in pairs] + ['_'.join((y, x)) for x, y in pairs]
            vocab.update(Counter(pairs))
            if iline % 10000 == 0:
                print(iline)
    
    vocab = {(pair, count) for pair, count in vocab.items() if count >= thr}
    save_count_vocabulary(vocab_file + '.counts', vocab)
    
    vocab = calc_pmi(vocab)
    vocab = {(pair, pmi) for pair, pmi in vocab if pmi >= pmi_thr}
    save_count_vocabulary(vocab_file + '.pmi', vocab)

    vocab = {pair for pair, pmi in vocab}
    save_vocabulary(vocab_file, vocab)


def calc_pmi(vocab):
    stats = [(tuple(map(int, pair.split('_'))), float(count)) for pair, count in vocab]
    
    counts = Counter()
    for (x, y), count in stats:
        counts[x] += count
    total = sum(counts.values())
    
    pmi = [('_'.join(map(str, (x, y))), log(count * total / (counts[x] * counts[y]))) for (x, y), count in stats]
    return pmi


if __name__ == '__main__':
    main()
