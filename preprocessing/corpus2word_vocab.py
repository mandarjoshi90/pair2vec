from collections import Counter
from math import sqrt
from random import Random

from docopt import docopt

from vocab import save_vocabulary, save_count_vocabulary


def main():
    args = docopt("""
    Usage:
        corpus2pairs.py [options] <corpus> <vocab>
    
    Options:
        --thr NUM    The minimal word count for being in the vocabulary [default: 100]
    """)
    
    corpus_file = args['<corpus>']
    vocab_file = args['<vocab>']
    thr = int(args['--thr'])
    
    vocab = Counter()
    with open(corpus_file) as f:
        for line in f:
            vocab.update(Counter(line.strip().lower().split()))
    vocab = sorted({(token, count) for token, count in vocab.items() if count >= thr}, key=lambda x: x[1], reverse=True)
    save_count_vocabulary(vocab_file + '.counts', vocab)
    vocab = [token for token, count in vocab]
    save_vocabulary(vocab_file, vocab)


if __name__ == '__main__':
    main()
