import torch
from collections import OrderedDict
import sys

def convert(old_file, new_file):
    items = list(torch.load(old_file).items())
    for i, (k, v) in enumerate(items):
        if k.startswith('relemb'):
            items[i]  = (k.replace('relemb', 'pair2vec'), v)
    torch.save(OrderedDict(items), new_file)

if __name__ == '__main__':
    convert(sys.argv[1], sys.argv[2])

