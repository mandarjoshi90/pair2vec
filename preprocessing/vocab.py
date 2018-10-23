

def save_vocabulary(path, vocab):
    with open(path, 'w') as f:
        for w in vocab:
            print(w, file=f)


def load_vocabulary(path):
    with open(path) as f:
        vocab = [line.strip() for line in f if len(line) > 0]
    return dict([(a, i) for i, a in enumerate(vocab)]), vocab


def save_count_vocabulary(path, vocab):
    with open(path, 'w') as f:
        for w, c in vocab:
            print(w, c, file=f)


def load_count_vocabulary(path):
    with open(path) as f:
        vocab = [line.strip().split() for line in f if len(line) > 0]
    return {w: int(c) for w, c in vocab}

