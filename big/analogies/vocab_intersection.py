import sys
import gzip
import fnmatch
import os

def read_vocab_file(fname):
    vocab = set()
    with open (fname, encoding='utf-8') as f:
        for line in f:
            vocab.add(line.strip())
    return vocab

def read_glove_vocab(fname, embedding_dim=300):
    vocab = set()
    with gzip.open(fname, 'rb') as embeddings_file:
        for line in embeddings_file:
            fields = line.decode('utf8').rstrip().split(' ')
            if len(fields) - 1 != embedding_dim:
                # Sometimes there are funny unicode parsing problems that lead to different
                # fields lengths (e.g., a word with a unicode space character that splits
                # into more than one column).  We skip those lines.  Note that if you have
                # some kind of long header, this could result in all of your lines getting
                # skipped.  It's hard to check for that here; you just have to look in the
                # embedding_misses_file and at the model summary to make sure things look
                # like they are supposed to.
                print("Found line with wrong number of dimensions (expected %d, was %d): %s", embedding_dim, len(fields) - 1, line)
                continue
            word = fields[0]
            vocab.add(word)
    return vocab

def get_bats_vocab(bats_dir):
    vocab = set()
    for root, dirnames, filenames in os.walk(bats_dir):
        for filename in fnmatch.filter(sorted(filenames), '[LE]*.txt'):
            with open(os.path.join(root, filename), encoding='utf-8') as f:
                id_line = 0
                for id_line, line in enumerate(f):
                    if "\t" in line:
                        left,right = line.lower().split("\t")
                    else:
                        left,right = line.lower().split()
                    right = right.strip()
                    if "/" in right:
                        right=[i.strip() for i in right.split("/")]
                    else:
                        right=[i.strip() for i in right.split(",")]
                    vocab.add(left)
                    for r in right:
                        vocab.add(r)
    return vocab

def intersection(embedding_file, vocab_file, bats_dir):
    vocab = read_vocab_file(vocab_file)
    glove_vocab = read_glove_vocab(embedding_file)
    bats_vocab = get_bats_vocab(bats_dir)
    print(len(vocab.intersection(glove_vocab)), len(vocab), len(glove_vocab))
    print(len(vocab.intersection(bats_vocab)), len(vocab), len(bats_vocab))
    print(len(glove_vocab.intersection(bats_vocab)), len(glove_vocab), len(bats_vocab))
    print(glove_vocab.intersection(bats_vocab) - vocab.intersection(bats_vocab))

def write_glove_vocab(fname, vocab_file):
    vocab = read_glove_vocab(fname)
    with open(vocab_file, mode='w', encoding='utf-8') as f:
        for word in vocab:
            f.write(word + '\n')


embedding_file = sys.argv[1]
vocab_file = sys.argv[2]
# bats_dir = sys.argv[3]
write_glove_vocab(embedding_file, vocab_file)
# intersection(embedding_file, vocab_file, bats_dir)
