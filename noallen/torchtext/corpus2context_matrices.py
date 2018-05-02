from collections import Counter
from math import sqrt
from random import Random
from docopt import docopt
import numpy as np
import os
from tqdm import tqdm
from noallen.torchtext.vocab import Vocab

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
    vocab_file = os.path.join(triplets_dir, 'vocab.txt')
    chunk = int(args['--chunk'])
    thr = int(args['--thr'])
    win = int(args['--win'])
    left = int(args['--left'])
    right = int(args['--right'])
    
    unk, pad, x_placeholder, y_placeholder = '<unk>', '<pad>', '<X>', '<Y>'
    print('reading vocab from {}'.format(vocab_file))
    vocab = get_vocab(vocab_file, corpus_file, [unk, pad, x_placeholder, y_placeholder])
    print('Vocab Size:', len(vocab))
    
    chunk_i = 1
    matrix = []
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for i_line, line in tqdm(enumerate(f)):
            tokens = line.strip().lower().split()
            token_ids = [vocab.stoi[t] for t in tokens if vocab.stoi[t] != 0]
            len_tokens = len(token_ids)
            
            for ix, x in enumerate(token_ids):
                y_iter = [iy for iy in range(ix + 1, ix + 2 + win) if iy < len_tokens and token_ids[iy] != 0]
               
                for iy in y_iter:
                    contexts = token_ids[max(0, ix - left): ix] + [vocab.stoi[x_placeholder]] + token_ids[ix+1: iy] + [vocab.stoi[y_placeholder]] + token_ids[iy+1:iy+right+1]
                    contexts += [vocab.stoi[pad]] * (left + right + win  + 2 - len(contexts))
                    matrix += [[token_ids[ix], token_ids[iy]] + contexts]
                    #import ipdb
                    #ipdb.set_trace()

            if (i_line + 1) % chunk == 0:
                #import ipdb
                #ipdb.set_trace()
                save(matrix, triplets_dir, chunk_i)
                matrix = []
                chunk_i += 1
                #if chunk_i >= 2:
                #    break
    
    if len(matrix) > 0:
        save(matrix, triplets_dir, chunk_i)

def get_vocab(vocab_path, corpus_file, specials):
    if os.path.isfile(vocab_path):
        vocab  = read_vocab_from_file(vocab_path, specials)
    else:
        selected = read_vocab(corpus_file)
        vocab = Vocab(selected, specials)
        save_vocab(selected, vocab_path)
    return vocab

def read_vocab_from_file(vocab_path, specials):
    tokens = None
    with open(vocab_path) as f:
        text = f.read()
        tokens = text.rstrip().split('\n')
    vocab = Vocab(tokens, specials=specials)
    return vocab

def read_vocab(corpus_file, thr=100, max_size=150000):
    counter = Counter()
    with open(corpus_file, mode='r', encoding='utf-8') as f:
        for i_line, line in enumerate(f):
            counter.update(Counter(line.strip().lower().split()))
            if i_line % 1000000 == 0:
                print(i_line)
    words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
    words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)
    selected = []
    for word, freq in words_and_frequencies:
        if freq < thr or len(selected) == max_size:
            break
        selected.append(word)
    return selected

def save_vocab(selected, path):
    with open(path, 'w', encoding='utf-8') as fout:
        fout.write('\n'.join(selected))


def save(matrix, triplets_dir, chunk_i):
    np.save(triplets_dir + '/triplets_' + str(chunk_i) + '.npy', np.array(tuple(matrix), dtype=np.int32))


if __name__ == '__main__':
    main()

