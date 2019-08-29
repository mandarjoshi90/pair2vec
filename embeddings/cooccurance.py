from collections import defaultdict
import sys
from tqdm import tqdm
from embeddings.vocab import Vocab

def read_vocab_from_file(vocab_path, specials):
    tokens = None
    with open(vocab_path) as f:
        text = f.read()
        tokens = text.rstrip().split('\n')
    tokens = tokens[3:]
    vocab = Vocab(tokens, specials=specials)
    print('Loaded vocab with {} tokens'.format(len(tokens)))
    return vocab

def get_cooccurance(fname, vocab_file, outf):
    vocab = read_vocab_from_file(vocab_file, specials=['<unk>', '<pad>', '<X>', '<Y>'])
    counts = defaultdict(int)
    win = 5
    with open(fname, encoding='utf-8') as f:
        for i_line, line in tqdm(enumerate(f)):
            tokens = line.strip().lower().split()
            token_ids = [vocab.stoi[t] for t in tokens if vocab.stoi[t] != 0]
            len_tokens = len(token_ids)
            for ix, x in enumerate(token_ids):
                y_iter = [iy for iy in range(ix + 1, ix + 2 + win) if iy < len_tokens and token_ids[iy] != 0]
                for iy in y_iter:
                    pair = (token_ids[ix], token_ids[iy]) if token_ids[ix] < token_ids[iy] else (token_ids[iy], token_ids[ix])
                    counts[pair] += 1
    counts = sorted(counts.items(), key=lambda x : x[1], reverse=True)
    with open(outf, mode='w', encoding='utf-8') as f:
        for pair, count in counts:
            f.write(vocab.itos[pair[0]] + '\t' + vocab.itos[pair[1]] + '\t' + str(count) + '\n')

corpus_file = sys.argv[1]  # input corput
pair_counts_file = sys.argv[2]  # output
vocab_file = sys.argv[3] if len(sys.argv) > 3 else 'vocabulary/pair2vec_tokens.txt' # optional vocaulary file from the repo
get_cooccurance(corpus_file, vocab_file, pair_counts_file)
#get_cooccurance('/sdb/data/wikipedia-sentences/shuf_sentences.txt', '/sdb/data/wikipedia-sentences/triplet_contexts/vocab.txt',
#        '/sdb/data/wikipedia-sentences/coor_counts.txt')
