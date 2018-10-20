import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def read(fname):
    f = open(fname)
    data = []
    for line in f:
        parts = line.strip().split('\t')
        parts = [part if part[0].isalpha() else float(part) for part in parts]
        data += [parts]
    return data

def replace(dic, key, replace):
    dic[replace] = dic[key]
    del dic[key]
    return dic


def aggregate(data):
    alpha = np.linspace(0, 1, 11)
    aplha = [float(a) for a in alpha]
    rel_dict = {'D': [0]*11, 'L': [0]*11, 'I': [0]*11, 'E': [0]*11}
    rel_total = {'D': [0]*11, 'L': [0]*11, 'I': [0]*11, 'E': [0]*11}
    for rel, correct, total, acc, a in data:
        ind = int(a * 10)
        rel_dict[rel[0]][ind] += correct
        rel_total[rel[0]][ind] += total
    rel_stats = {'alpha': [1.0-a for a in alpha]}
    for rel in rel_dict.keys():
        acc = [corr * 100.0 / total for corr, total in zip(rel_dict[rel], rel_total[rel])]
        rel_stats[rel] = acc
    rel_stats = replace(rel_stats, 'D', 'Derivational')
    rel_stats = replace(rel_stats, 'L', 'Lexicographic')
    rel_stats = replace(rel_stats, 'I', 'Inflectional')
    rel_stats = replace(rel_stats, 'E', 'Encyclopedic')
    return rel_stats


def create_dlei_fig(dataf, save_dir):
    data = read(dataf)
    rel_accs = aggregate(data)
    df = pd.DataFrame(rel_accs)
    # import ipdb
    # ipdb.set_trace()
    plt.plot( 'alpha', 'Derivational', data=df, marker='', color='gold', linewidth=2)
    plt.plot( 'alpha', 'Lexicographic', data=df, marker='', color='firebrick', linewidth=2)
    plt.plot( 'alpha', 'Inflectional', data=df, marker='', color='g', linewidth=2)
    plt.plot( 'alpha', 'Encyclopedic', data=df, marker='', color='dodgerblue', linewidth=2)
    plt.xlabel(r'$\alpha$')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 100.0)
    keys = list(rel_accs)
    keys.remove('alpha')
    plt.legend(tuple(keys),
           loc='upper right', shadow=True)
    plt.savefig(os.path.join(save_dir, 'dlie.png'))

if __name__ == '__main__':
    dataf = sys.argv[1]
    save_dir = sys.argv[2]
    create_dlei_fig(dataf, save_dir)
