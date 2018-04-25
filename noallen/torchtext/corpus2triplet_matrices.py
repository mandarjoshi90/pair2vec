import codecs
from collections import Counter
from math import sqrt
from random import Random
from docopt import docopt
import numpy as np


def main():
    args = docopt("""
    Usage:
        corpus2triplet_matrices.py [options] <corpus> <triplets_dir>
    
    Options:
        --chunk NUM    The number of lines to read before dumping each matrix [default: 100000]
        --thr NUM      The minimal word count for being in the vocabulary [default: 1000]
        --win NUM      Maximal number of tokens between X and Y [default: 5]
        --left NUM     Left window size [default: 2]
        --right NUM    Right window size [default: 1]
        --compressed   Whether to compress the contexts of each X,Y pair into one row
    """)
    
    corpus_file = args['<corpus>']
    triplets_dir = args['<triplets_dir>']
    chunk = int(args['--chunk'])
    thr = int(args['--thr'])
    win = int(args['--win'])
    left = int(args['--left'])
    right = int(args['--right'])
    
    wi, iw = read_vocab(corpus_file, thr)
    print 'Vocab Size:', len(iw)
    save_vocab(iw, triplets_dir)
    
    L = 1*len(iw)
    R = 2*len(iw)

    chunk_i = 1
    matrix = []
    with codecs.open(corpus_file, 'r', 'utf-8') as f:
        for i_line, line in enumerate(f):
            tokens = [t if t in wi else None for t in line.strip().lower().split()]
            len_tokens = len(tokens)
            
            x_iter = [(ix, wi[x]) for ix, x in enumerate(tokens) if x is not None]
            for ix, x in x_iter:
                y_iter = [(iy, wi[tokens[iy]]) for iy in xrange(ix + 1, ix + 2 + win) if iy < len_tokens and tokens[iy] is not None]
                for iy, y in y_iter:
                    if iy == ix + 1:
                        contexts = [(wi[''])]
                    else:
                        contexts = [(wi[tokens[ir]]) for ir in xrange(ix + 1, iy) if tokens[ir] is not None]
                    contexts.extend([(wi[tokens[ir]] + L) for ir in xrange(ix - left, ix) if ir >= 0 and tokens[ir] is not None])
                    contexts.extend([(wi[tokens[ir]] + R) for ir in xrange(iy + 1, iy + 1 + right) if ir < len_tokens and tokens[ir] is not None])
                    
                    matrix.extend([(x, y, r) for r in contexts])

            if (i_line + 1) % chunk == 0:
                save(matrix, triplets_dir, chunk_i)
                matrix = []
                chunk_i += 1
    
    save(matrix, triplets_dir, chunk_i)


def read_vocab(corpus_file, thr):
    vocab = Counter()
    with codecs.open(corpus_file, 'r', 'utf-8') as f:
        for i_line, line in enumerate(f):
            vocab.update(Counter(line.strip().lower().split()))
            if i_line % 1000000 == 0:
                print i_line
    
    iw = list(set([''] + [token for token, count in vocab.items() if count >= thr]))
    wi = {w: i for i, w in enumerate(iw)}
    return wi, iw


def save_vocab(vocab, triplets_dir):
    with codecs.open(triplets_dir + '/vocab.txt', 'w', 'utf-8') as fout:
        print >>fout, '\n'.join(vocab)


def save(matrix, triplets_dir, chunk_i):
    np.save(triplets_dir + '/triplets_' + str(chunk_i) + '.npy', np.array(tuple(matrix), dtype=np.int32))


if __name__ == '__main__':
    main()

