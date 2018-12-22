from collections import Counter
from math import sqrt
from random import Random
from docopt import docopt
import numpy as np
import os
from tqdm import tqdm
from embeddings.vocab import Vocab
import pickle
from collections import defaultdict

stop_words = set(['the', 'of', ',', 'in', 'and', 'to', '"', '(', ')', 'a', 'is', 'was', 'for', '.', '-', 'as', 'by', 'at', 'an', 'with', 'from', 'that', 'which', 'also', 'be', 'were', 'are', 'but', 'this', 'had', 'can', 'into', 'could', 'would', 'should', 'then', 'do', 'does', 'above', 'after', 'again', 'same', 'any', 'been'])
def read_filtered_pairs(fname, vocab, thr=None, sorted_file=False):
    pairs_count = {}
    count = 1
    with open(fname, encoding='utf-8') as f:
        for line in tqdm(f):
            w1, w2, count = line.strip().split('\t')
            if thr is None or float(count) > thr:
                this_pair = (vocab.stoi[w1],vocab.stoi[w2]) if vocab.stoi[w1] < vocab.stoi[w2] else (vocab.stoi[w2],vocab.stoi[w1])
                pairs_count[this_pair] =float(count)
            elif sorted_file:
                break
    total = float(sum(pairs_count.values()))
    for k, count in pairs_count.items():
        pairs_count[k] /= total
    return pairs_count


def read_counts(fname, vocab, thr=10):
    count_dict = defaultdict(int)
    with open(fname, encoding='utf-8') as f:
        for line in tqdm(f):
            w1, count = line.strip().split('\t')
            if int(count) > thr and w1 in vocab.stoi:
                count_dict[vocab.stoi[w1]] =float(count)
    total = float(sum(count_dict.values()))
    for k, count in count_dict.items():
        count_dict[k] /= total
    # print('total {}, min {}'.format(total, thr / total))
    return count_dict


def main():
    args = docopt("""
    Usage:
        preprocess.py [options] <corpus> <triplets_dir> <word_count_file> <pair_count_file>

    Options:
        --chunk NUM         The number of lines to read before dumping each matrix [default: 1000000]
        --win NUM           Maximal number of tokens between X and Y [default: 4]
        --left NUM          Left window size [default: 1]
        --right NUM         Right window size [default: 1]
        --word_thr NUM      Right window size [default: 10]
        --pair_thr NUM      Right window size [default: 50]
    """)
    print(args)
    corpus_file = args['<corpus>']
    triplets_dir = args['<triplets_dir>']
    word_count_file = args['<word_count_file>']
    pair_count_file = args['<pair_count_file>']
    vocab_file = os.path.join(triplets_dir, 'vocab.txt')
    word_thr = int(args['--word_thr'])
    pair_thr = int(args['--pair_thr'])
    chunk = int(args['--chunk'])
    win = int(args['--win'])
    left = int(args['--left'])
    right = int(args['--right'])
    unk, pad, x_placeholder, y_placeholder = '<unk>', '<pad>', '<X>', '<Y>'
    print('reading vocab from {}'.format(vocab_file))
    specials  = [unk, pad, x_placeholder, y_placeholder]
    vocab = get_vocab(vocab_file, corpus_file, specials)
    print('Vocab Size:', len(vocab))
    chunk_i = 1
    matrix = []
    pair_filter = read_filtered_pairs(pair_count_file, vocab, pair_thr, sorted_file=True)
    stop_word_ids = set([vocab.stoi[w] for w in stop_words])
    keep_wordpair = keep_wordpair_by_mult
    word_unigram_dict = read_counts(word_count_file, vocab, word_thr)

    with open(corpus_file, 'r', encoding='utf-8') as f:
        for i_line, line in tqdm(enumerate(f)):
            tokens = line.strip().lower().split()
            token_ids = [vocab.stoi[t] for t in tokens if vocab.stoi[t] != 0]
            len_tokens = len(token_ids)
            for ix, x in enumerate(token_ids):
                # use ix+1 to start from adjacent word
                y_start = (ix + 1)
                y_iter = [iy for iy in range(y_start, ix + 2 + win) if iy < len_tokens and token_ids[iy] != 0]
                for iy in y_iter:
                    ordered_pair = (token_ids[ix], token_ids[iy])
                    this_pair = (token_ids[ix], token_ids[iy]) if (token_ids[ix] < token_ids[iy]) else (token_ids[iy], token_ids[ix])

                    if this_pair in pair_filter and keep_wordpair(word_unigram_dict, this_pair, vocab, stop_words=stop_word_ids):
                        contexts = token_ids[max(0, ix - left): ix] + [vocab.stoi[x_placeholder]] + token_ids[ix+1: iy] + [vocab.stoi[y_placeholder]] + token_ids[iy+1:iy+right+1]
                        contexts += [vocab.stoi[pad]] * (left + right + win  + 2 - len(contexts))
                        matrix += [[token_ids[ix], token_ids[iy]] + contexts]

            if (i_line + 1) % chunk == 0:
                size = len(matrix)
                save(matrix, triplets_dir, chunk_i)
                print('chunk {} len {}'.format(chunk_i, len(matrix)))
                matrix = []
                chunk_i += 1
    if len(matrix) > 0:
        save(matrix, triplets_dir, chunk_i)


def keep_wordpair_by_mult(count_dict, word_pair, vocab, thr=5e-5, stop_words=None):
    x, y = word_pair
    clamp = lambda x : x if x < 1.0 else 1.0
    keep_x = clamp(sqrt(thr / count_dict[x])) if x in count_dict else 0.0
    keep_y = clamp(sqrt(thr / count_dict[y])) if y in count_dict else 0.0
    random_prob = np.random.uniform()
    return  random_prob < keep_x * keep_y

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

