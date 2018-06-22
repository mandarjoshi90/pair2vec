from collections import Counter
from math import sqrt
from random import Random
from docopt import docopt
import numpy as np
import os
from tqdm import tqdm
from noallen.torchtext.vocab import Vocab
import pickle
from collections import defaultdict

stop_words = set(['the', 'of', ',', 'in', 'and', 'to', '"', '(', ')', 'a', 'is', 'was', 'for', '.', '-', 'as', 'by', 'at', 'an', 'with', 'from', 'that', 'which', 'also', 'be', 'were', 'are', 'but', 'this', 'had', 'can', 'into'])
def read_filtered_pairs(fname, vocab, thr=None, sorted_file=False):
    pairs = set()
    with open(fname, encoding='utf-8') as f:
        for line in tqdm(f):
            w1, w2, count= line.strip().split('\t')
            # w1, _, w2 = line.strip().split('\t')
            if w1 in stop_words or w2 in stop_words:
                continue
            if thr is None or float(count) > thr:
                this_pair = (vocab.stoi[w1],vocab.stoi[w2]) if vocab.stoi[w1] < vocab.stoi[w2] else (vocab.stoi[w2],vocab.stoi[w1])
                pairs.add(this_pair)
            elif sorted_file:
                break
    return pairs

def read_pair_vocab(fname, vocab):
    words = set()
    with open(fname, encoding='utf-8') as f:
        for line in tqdm(f):
            w = line.strip()
            words.add(vocab.stoi[w])
    return words

def main():
    args = docopt("""
    Usage:
        corpus2triplet_matrices.py [options] <corpus> <triplets_dir> 
    
    Options:
        --reverse NUM       whether to add the reversed context [default: 0]
        --pair_vocab NUM    File containing pair vocab
        --chunk NUM         The number of lines to read before dumping each matrix [default: 100000]
        --thr NUM           The minimal word count for being in the vocabulary [default: 1000]
        --win NUM           Maximal number of tokens between X and Y [default: 5]
        --left NUM          Left window size [default: 2]
        --right NUM         Right window size [default: 1]
        --gran NUM          Single instance per word (0), per word-position (1), full pattern(2) [default: 0]
        --skip-next NUM     Skip next word pattern [default: 0]
    """)
    print(args)
    corpus_file = args['<corpus>']
    triplets_dir = args['<triplets_dir>']
    vocab_file = os.path.join(triplets_dir, 'vocab.txt')
    chunk = int(args['--chunk'])
    reverse = bool(int(args['--reverse']))
    thr = int(args['--thr'])
    win = int(args['--win'])
    left = int(args['--left'])
    right = int(args['--right'])
    granularity = int(args['--gran'])
    skip_next = int(args['--skip-next']) == 1
    pair_vocab = str(args['--pair_vocab']) if args['--pair_vocab'] is not None else None
    print('granularity {} skip_next {} reverse {}'.format(granularity, skip_next, reverse))
    
    unk, pad, x_placeholder, y_placeholder, blank = '<unk>', '<pad>', '<X>', '<Y>', ''
    print('reading vocab from {}'.format(vocab_file))
    specials  = [unk, blank] if granularity < 3 else  [unk, pad, x_placeholder, y_placeholder]
    vocab = get_vocab(vocab_file, corpus_file, specials)
    pair_vocab = read_pair_vocab(pair_vocab, vocab) if pair_vocab is not None else None
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
    coor_filter = read_filtered_pairs('/sdb/data/wikipedia-sentences/sorted_coor_counts.txt', vocab, 50, sorted_file=True)
    #coor_filter = read_filtered_pairs('/home/mandar90/data/conceptnet/single_word/train.txt', vocab, None, sorted_file=False)
    print('count filter {}'.format(len(coor_filter)))
    # glove_filter = read_filtered_pairs('/sdb/data/wikipedia-sentences/pairs.txt', vocab, 0.4)
    # print('glove filter {}'.format(len(glove_filter)))
    filtered_pairs = None #lexinf_filter #.union(glove_filter)
    # print('final filter {}'.format(len(filtered_pairs)))
    pair_to_index = defaultdict()
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
                    ordered_pair = (token_ids[ix], token_ids[iy])
                    this_pair = (token_ids[ix], token_ids[iy]) if (token_ids[ix] < token_ids[iy]) else (token_ids[iy], token_ids[ix])
                    # if this_pair in coor_filter and (token_ids[ix] in pair_vocab or token_ids[iy] in pair_vocab):
                    # if token_ids[ix] in pair_vocab or token_ids[iy] in pair_vocab:
                    if this_pair in coor_filter:
                        # if pair_vocab is not None:
                        # pair_to_index[ordered_pair] = pair_to_index.get(ordered_pair, len(pair_to_index))
                        if granularity == 3:
                            contexts = token_ids[max(0, ix - left): ix] + [vocab.stoi[x_placeholder]] + token_ids[ix+1: iy] + [vocab.stoi[y_placeholder]] + token_ids[iy+1:iy+right+1]
                            contexts += [vocab.stoi[pad]] * (left + right + win  + 2 - len(contexts))
                            # contexts =   token_ids[ix+1: iy]
                            # contexts += [vocab.stoi[pad]] * ( win  - len(contexts))
                            matrix += [[token_ids[ix], token_ids[iy]] + contexts]
                            if reverse:
                                contexts = token_ids[max(0, ix - left): ix] + [vocab.stoi[y_placeholder]] + token_ids[ix+1: iy] + [vocab.stoi[x_placeholder]] + token_ids[iy+1:iy+right+1]
                                contexts += [vocab.stoi[pad]] * (left + right + win  + 2 - len(contexts))
                                matrix += [[token_ids[iy], token_ids[ix]] + contexts]
                            # matrix += [[pair_to_index[ordered_pair]] + contexts]
                        elif granularity == 2:
                            left_ctx = token_ids[ix - 1] if ix > 0 else vocab.stoi[blank]
                            right_ctx = token_ids[iy + 1] if iy < (len(token_ids) - 1) else vocab.stoi[blank]
                            ctx_pos =  list(range(ix + 1, iy)) 
                            contexts = [[token_ids[ix], token_ids[iy], left_ctx, right_ctx, token_ids[pos] + len(vocab) * positions_dict[(0, iy-ix, pos - ix)]] for pos in ctx_pos]
                            if len(contexts) == 0 and ix + 1  == iy:
                                contexts = [[token_ids[ix], token_ids[iy], left_ctx, right_ctx, vocab.stoi[blank] + len(vocab) * positions_dict[(0, 1, 0)]]]
                            matrix += contexts
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
                size = len(matrix)
                save(matrix, triplets_dir, chunk_i)
                # sample_and_save(matrix, triplets_dir, chunk_i, vocab)
                print('chunk {} len {} pairs {}'.format(chunk_i, len(matrix), len(pair_to_index)))
                matrix = []
                chunk_i += 1
                # import ipdb
                # ipdb.set_trace()
                # if chunk_i >= 50:
                    # break
    # pickle.dump(pair_to_index, open('pair_to_index.pkl', 'wb'))
    print("Num_pairs {}".format(len(pair_to_index)))
    if len(matrix) > 0:
        # save(matrix, triplets_dir, chunk_i)
        save(matrix, triplets_dir, chunk_i, vocab)

def sample_and_save(matrix, triplets_dir, chunk_i, vocab, thr=100):
    matrix = np.array(matrix)
    sub, obj = np.copy(matrix[:, 0]), np.copy(matrix[:, 1])
    np.random.shuffle(sub)
    np.random.shuffle(obj)
    rel_to_sub, rel_to_obj = defaultdict(list), defaultdict(list)
    relp_to_sub, relp_to_obj = defaultdict(list), defaultdict(list)
    for instance in matrix:
        contexts = instance[2:]
        rel = [contexts[idx] for idx in range(1, len(contexts)) if contexts[idx] not in range(0, 4)]
        rel_to_sub[tuple(contexts)].append(instance[0])
        rel_to_obj[tuple(contexts)].append(instance[1])
        relp = ' '.join([vocab.itos[idx] for idx in rel])
        relp_to_sub[relp].append(vocab.itos[instance[0]])
        relp_to_obj[relp].append(vocab.itos[instance[1]])
    import ipdb
    ipdb.set_trace()
    sampled_subjects, sampled_objects = [], []
    for i, instance in enumerate(matrix):
        contexts = instance[2:]
        rel_subs, rel_objs = rel_to_sub[tuple(contexts)], rel_to_obj[tuple(contexts)]
        sampled_sub = (np.random.choice(rel_subs)) if len(rel_subs) > thr else sub[i]
        sampled_obj = (np.random.choice(rel_objs)) if len(rel_objs) > thr else obj[i]
        # if len(rel_subs) > 50:
            # print(len(rel_subs), len(set(rel_subs)), [vocab.itos[i] for i in contexts])
        sampled_subjects.append([sampled_sub])
        sampled_objects.append([sampled_obj])
    sampled_subjects = np.array(sampled_subjects)
    sampled_objects = np.array(sampled_objects)
    matrix = np.concatenate((sampled_subjects, sampled_objects, matrix), axis=1)
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

