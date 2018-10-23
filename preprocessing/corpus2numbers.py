from collections import Counter
from math import sqrt
from random import Random

from docopt import docopt

from vocab import load_vocabulary


def main():
    args = docopt("""
    Usage:
        corpus2numbers.py <corpus> <vocab>
    """)
    
    corpus_file = args['<corpus>']
    vocab_file = args['<vocab>']
    
    wi, iw = load_vocabulary(vocab_file)
    
    with open(corpus_file) as f: 
        for line in f:
            tokens = [wi[t] if t in wi else -1 for t in line.strip().lower().split()]
            print(' '.join(map(str, tokens)))


if __name__ == '__main__':
    main()
