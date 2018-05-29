from collections import Counter
from math import sqrt
from random import Random
from docopt import docopt
import numpy as np
import os
from tqdm import tqdm
from noallen.torchtext.vocab import Vocab
def read_filtered_pairs(fname, vocab, thr=None, sorted_file=False):
    pairs = set()
    with open(fname, encoding='utf-8') as f:
        for line in tqdm(f):
            w1, w2, count= line.strip().split('\t')
            if thr is None or float(count) > thr:
                this_pair = (vocab.stoi[w1],vocab.stoi[w2]) if vocab.stoi[w1] < vocab.stoi[w2] else (vocab.stoi[w2],vocab.stoi[w1])
                pairs.add(this_pair)
            elif sorted_file:
                break
    return pairs

def main():
    args = docopt("""
    Usage:
        corpus2triplet_matrices.py [options] <corpus> <triplets_dir> 
    
    Options:
        --chunk NUM         The number of lines to read before dumping each matrix [default: 100000]
        --thr NUM           The minimal word count for being in the vocabulary [default: 1000]
        --win NUM           Maximal number of tokens between X and Y [default: 5]
        --left NUM          Left window size [default: 2]
        --right NUM         Right window size [default: 1]
        --gran NUM          Single instance per word (0), per word-position (1), full pattern(2) [default: 0]
        --skip-next NUM     Skip next word pattern [default: 1]
    """)
    print(args)
    corpus_file = args['<corpus>']
    triplets_dir = args['<triplets_dir>']
    vocab_file = os.path.join(triplets_dir, 'vocab.txt')
    chunk = int(args['--chunk'])
    thr = int(args['--thr'])
    win = int(args['--win'])
    left = int(args['--left'])
    right = int(args['--right'])
    granularity = int(args['--gran'])
    skip_next = int(args['--skip-next']) == 1
    print('granularity {} skip_next {}'.format(granularity, skip_next))
    
    unk, pad, x_placeholder, y_placeholder, blank = '<unk>', '<pad>', '<X>', '<Y>', ''
    print('reading vocab from {}'.format(vocab_file))
    specials  = [unk, blank] if granularity < 3 else  [unk, pad, x_placeholder, y_placeholder]
    vocab = get_vocab(vocab_file, corpus_file, specials)
    print('Vocab Size:', len(vocab))
    if granularity == 1:
        positions = [(0, ypos, rpos) for ypos in range(1, win + 2) for rpos in list(range(-left, 0)) + list(range(1, ypos)) + list(range(ypos+1, ypos + right + 1)) ]
    elif granularity == 2:
        positions = [(0, ypos, rpos) for ypos in range(1, win + 2) for rpos in list(range(1, ypos)) ] + [(0,1,0)]
    else:
        positions = []
    positions_dict = {pos : i for i, pos in enumerate(positions)}
    print("len of pos dict {}".format(len(positions_dict)))
    
    chunk_i = 1
    matrix = []
    L = 1*len(vocab)
    R = 2*len(vocab)
    # lexinf_filter = read_filtered_pairs('/home/mandar90/data/lexinf/root09/train.txt', vocab)
    # print('lexinf_filter {}'.format(len(lexinf_filter)))
    # coor_filter = read_filtered_pairs('/sdb/data/wikipedia-sentences/sorted_coor_counts.txt', vocab, 200, sorted_file=True)
    # print('count filter {}'.format(len(coor_filter)))
    # glove_filter = read_filtered_pairs('/sdb/data/wikipedia-sentences/pairs.txt', vocab, 0.4)
    # print('glove filter {}'.format(len(glove_filter)))
    filtered_pairs = None #lexinf_filter #.union(glove_filter)
    # print('final filter {}'.format(len(filtered_pairs)))
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for i_line, line in tqdm(enumerate(f)):
            tokens = line.strip().lower().split()
            token_ids = [vocab.stoi[t] for t in tokens if vocab.stoi[t] != 0]
            len_tokens = len(token_ids)
            
            for ix, x in enumerate(token_ids):
                # use ix+1 to start from adjacent word
                y_start = (ix + 1) if not skip_next else (ix + 2)
                y_iter = [iy for iy in range(y_start, ix + 2 + win) if iy < len_tokens and token_ids[iy] != 0]
               
                for iy in y_iter:
                    this_pair = (token_ids[ix], token_ids[iy]) if (token_ids[ix] < token_ids[iy]) else (token_ids[iy], token_ids[ix])
                    if filtered_pairs is None or this_pair in filtered_pairs:
                        if granularity == 3:
                            contexts = token_ids[max(0, ix - left): ix] + [vocab.stoi[x_placeholder]] + token_ids[ix+1: iy] + [vocab.stoi[y_placeholder]] + token_ids[iy+1:iy+right+1]
                            contexts += [vocab.stoi[pad]] * (left + right + win  + 2 - len(contexts))
                            # contexts =   token_ids[ix+1: iy]
                            # contexts += [vocab.stoi[pad]] * ( win  - len(contexts))
                            matrix += [[token_ids[ix], token_ids[iy]] + contexts]
                        elif granularity == 2:
                            left_ctx = token_ids[ix - 1] if ix > 0 else vocab.stoi[blank]
                            right_ctx = token_ids[iy + 1] if iy < (len(token_ids) - 1) else vocab.stoi[blank]
                            ctx_pos =  list(range(ix + 1, iy)) 
                            contexts = [[token_ids[ix], token_ids[iy], left_ctx, right_ctx, token_ids[pos] + len(vocab) * positions_dict[(0, iy-ix, pos - ix)]] for pos in ctx_pos]
                            if len(contexts) == 0 and ix + 1  == iy:
                                contexts = [[token_ids[ix], token_ids[iy], left_ctx, right_ctx, vocab.stoi[blank] + len(vocab) * positions_dict[(0, 1, 0)]]]
                            matrix += contexts
                            # import ipdb
                            # ipdb.set_trace()
                        elif granularity == 1:
                            ctx_pos = list(range(max(0, ix - left) , ix)) + list(range(ix + 1, iy)) + list(range(min(len(token_ids) , iy+1), min(len(token_ids), iy+right+1)))
                            contexts = [[token_ids[ix], token_ids[iy], token_ids[pos] + len(vocab) * positions_dict[(0, iy-ix, pos - ix)]] for pos in ctx_pos]
                            matrix += contexts
                        else:
                            if iy == ix + 1:
                                contexts = [vocab.stoi[blank]]
                            else:
                                contexts = [token_ids[ir] for ir in range(ix + 1, iy) if tokens[ir]]
                            contexts.extend([(token_ids[ir] + L) for ir in range(ix - left, ix) if ir >= 0])
                            contexts.extend([(token_ids[ir] + R) for ir in range(iy + 1, iy + 1 + right) if ir < len(token_ids)])
                            matrix.extend([(token_ids[ix], token_ids[iy], r) for r in contexts])

            if (i_line + 1) % chunk == 0:
                save(matrix, triplets_dir, chunk_i)
                print('chunk {} len {}'.format(chunk_i, len(matrix)))
                matrix = []
                chunk_i += 1
                # if chunk_i >= 100:
                    # break
    
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

