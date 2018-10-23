import numpy as np
import sys


def save(data, path, chunk):
    np.save(path + str(chunk) + '.npy', np.array(tuple(data), np.int32))


max_len = 12 + 1 + 2

data_path = sys.argv[1]
out_path = sys.argv[2]

chunk = 0
i = 0
data = []
pad = [0] * (max_len - 5)
with open(data_path) as fin:
    for line in fin:
        pair, context = line.strip().split('\t')
        example = pair.split('_') + context.split(' ')
        example = [1 if w == '-1' else (2 if w == 'X' else (3 if w == 'Y' else int(w) + 4)) for w in example]
        example = (example + pad)[:max_len]
        data.append(tuple(example))
        i += 1
        if i % 10000000 == 0:
            save(data, out_path, chunk)
            data = []
            chunk += 1
    save(data, out_path, chunk)

