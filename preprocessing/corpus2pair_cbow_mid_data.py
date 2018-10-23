from collections import Counter
from math import sqrt
from random import Random

from docopt import docopt

from vocab import load_vocabulary


def main():
    args = docopt("""
    Usage:
        corpus2pair_context_data.py [options] <corpus> <vocab>

    Options:
        --middle NUM   The maximal token distance between the two target words [default: 12]
    """)
    
    corpus_file = args['<corpus>']
    vocab_file = args['<vocab>']
    middle_window = int(args['--middle'])
    
    vocab, _ = load_vocabulary(vocab_file)

    with open(corpus_file) as f:
        for line in f:
            tokens = list(line.strip().split())
            pairs = [('_'.join((x, y)), '_'.join((y, x)), ix, ix+1+iy) for ix, x in enumerate(tokens[:-1]) for iy, y in enumerate(tokens[ix+1:])]
            pairs = [(xy, yx, ix, iy) for xy, yx, ix, iy in pairs if xy in vocab and 2 <= iy - ix <= middle_window]
            examples = []
            for xy, yx, ix, iy in pairs:
                mcontexts = [tokens[i] for i in range(ix+1, iy)]
                xycontexts = ['X'] + mcontexts + ['Y']
                yxcontexts = ['Y'] + mcontexts + ['X']
                examples.append('\t'.join((xy, ' '.join(xycontexts))))
                examples.append('\t'.join((yx, ' '.join(yxcontexts))))
            output = '\n'.join(examples)
            if len(output) > 0:
                print(output)


if __name__ == '__main__':
    main()
