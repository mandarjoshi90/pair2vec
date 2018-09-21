import os
import sys
import torch
from torch.autograd import Variable
from torch.nn.functional import normalize
from noallen.torchtext.matrix_data import create_vocab
from noallen.torchtext.indexed_field import Field
from noallen.util import load_model, get_config
from noallen.model import RelationalEmbeddingModel, PairwiseRelationalEmbeddingModel, Pair2RelModel
import numpy as np
from collections import defaultdict
def get_pattern_representation(str_patterns, model):
    patterns = []
    for str_pattern in str_patterns:
        words = str_pattern.split(' ')
        words = words + ['<pad>'] * (8 - len(words))
        pattern = [model.rel_vocab.stoi[word] for word in words]
        pattern = Variable(torch.cuda.LongTensor(pattern), requires_grad=False)
        patterns.append(pattern)
    patterns = torch.stack(patterns, dim=0)
    mask = 1.0 - torch.eq(patterns, model.rel_vocab.stoi['<pad>']).float()
    pattern_representation = model.represent_relations((patterns, mask))
    print(torch.norm(pattern_representation, dim=1))
    return normalize(pattern_representation, dim=1)
    # return pattern_representation

def get_relation_embedding(word1, word2, model):
    word1_embed = model.represent_arguments(word1)
    word2_embed = model.represent_arguments(word2)
    relation_embedding = normalize(model.predict_relations(word1_embed, word2_embed), dim=1)
    return relation_embedding


def get_kbc_pairs(relation_pattern_file, model):
    word1, word2 = [], []
    with open(relation_pattern_file, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            print(parts)
            for i in range(4, len(model.arg_vocab)):
                word1.append(parts[1].strip())
                word2.append(model.arg_vocab.itos[i])
            # word1.append(parts[1].strip())
            # word2.append(parts[2].strip())
    return word1, word2

def get_word_pairs(relation_pattern_file):
    word1, word2 = [], []
    with open(relation_pattern_file, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            print(parts)
            word1.append(parts[1].strip())
            word2.append(parts[2].strip())
    return word1, word2


def word_pair_to_pairs(relation_instance_file, model):
    word1_str, word2_str = get_word_pairs(relation_instance_file)
    word1 = torch.cuda.LongTensor([model.arg_vocab.stoi[w] for w in word1_str])
    word2 = torch.cuda.LongTensor([model.arg_vocab.stoi[w] for w in word2_str])
    relation_embedding = get_relation_embedding(word1, word2, model)
    scores = torch.mm(relation_embedding, relation_embedding.transpose(0,1))
    scores = scores * Variable(1.0 - torch.eye(scores.size(0)).cuda().float(), requires_grad=False)
    sorted_scores, indices = torch.topk(scores, 5, dim=1)
    sorted_scores, indices = sorted_scores.cpu().data.numpy(), indices.cpu().data.numpy()
    for i in range(indices.shape[0]):
        print(word1_str[i], word2_str[i], [(word1_str[ind], word2_str[ind], score) for ind, score in zip(indices[i], sorted_scores[i])])
        print()

def word_pair_to_patterns(relation_pattern_file, relation_instance_file, model):
    str_patterns = get_patterns(relation_pattern_file)
    patterns = get_pattern_representation(str_patterns, model)
    word1_str, word2_str = get_word_pairs(relation_instance_file)
    word1 = torch.cuda.LongTensor([model.arg_vocab.stoi[w] for w in word1_str])
    word2 = torch.cuda.LongTensor([model.arg_vocab.stoi[w] for w in word2_str])
    relation_embedding = get_relation_embedding(word1, word2, model)
    scores = torch.mm(relation_embedding, patterns.transpose(0,1))
    sorted_scores, indices = torch.topk(scores, 50, dim=1)
    sorted_scores, indices = sorted_scores.cpu().data.numpy(), indices.cpu().data.numpy()
    for i in range(indices.shape[0]):
        print(word1_str[i], word2_str[i], [(str_patterns[ind], score) for ind, score in zip(indices[i], sorted_scores[i])])
        print()

def patterns_to_pairs(relation_pattern_file, relation_instance_file, model, neg_instance_file=None):
    str_patterns = get_patterns(relation_pattern_file)
    patterns = get_pattern_representation(str_patterns, model)
    word1_str, word2_str = get_word_pairs(relation_instance_file)
    # word1_str, word2_str = get_kbc_pairs(relation_instance_file, model)
    if neg_instance_file is not None:
        neg_word1_str, neg_word2_str = get_word_pairs(neg_instance_file)
        word1_str += neg_word1_str
        word2_str += neg_word2_str
    word1 = torch.cuda.LongTensor([model.arg_vocab.stoi[w] for w in word1_str])
    word2 = torch.cuda.LongTensor([model.arg_vocab.stoi[w] for w in word2_str])
    relation_embedding = get_relation_embedding(word1, word2, model)
    scores = torch.mm(patterns, relation_embedding.transpose(0,1))
    sorted_scores, indices = torch.topk(scores, 15, dim=1)
    sorted_scores, indices = sorted_scores.cpu().data.numpy(), indices.cpu().data.numpy()
    for i in range(indices.shape[0]):
        print(str_patterns[i], [(word1_str[ind], word2_str[ind], score) for ind, score in zip(indices[i], sorted_scores[i])] )
        print()
    pass

def get_patterns_to_pairs(pattern_file, instance_file):
    rel_to_patterns = defaultdict(list)
    with open(pattern_file, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            rel_to_patterns[parts[0]] += (parts[1:])
    rel_to_pairs = defaultdict(list)
    word1, word2 = [], []
    with open(instance_file, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            rel_to_pairs[parts[0]].append((parts[1], parts[2]))
    return rel_to_pairs, rel_to_patterns


def patterns_to_kbc(relation_pattern_file, relation_instance_file, model, neg_instance_file=None):
    rel_to_pairs, rel_to_patterns = get_patterns_to_pairs(relation_pattern_file, relation_instance_file)
    for relation, pairs in rel_to_pairs.items():
        str_patterns = rel_to_patterns[relation]
        patterns = get_pattern_representation(str_patterns, model)
        for w1, w2 in pairs:
            rep_w2 = [model.arg_vocab.itos[i] for i in range(4, len(model.arg_vocab))]
            rep_w1 = [w1] * len(rep_w2)
            word1 = torch.cuda.LongTensor([model.arg_vocab.stoi[w] for w in rep_w1])
            word2 = torch.cuda.LongTensor([model.arg_vocab.stoi[w] for w in rep_w2])
            relation_embedding = get_relation_embedding(word1, word2, model)
            scores = torch.mm(patterns, relation_embedding.transpose(0,1))
            sorted_scores, indices = torch.topk(scores, 15, dim=1)
            sorted_scores, indices = sorted_scores.cpu().data.numpy(), indices.cpu().data.numpy()
            # import ipdb
            # ipdb.set_trace()
            for i in range(indices.shape[0]):
                print(str_patterns[i], [(rep_w1[ind], rep_w2[ind], score) for ind, score in zip(indices[i], sorted_scores[i])] )
                print()

def get_patterns(relation_pattern_file):
    patterns = []
    with open(relation_pattern_file, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            for part in parts[1:]:
                patterns.append(part.strip())
    return patterns

def pattern_to_patterns(relation_pattern_file, model):
    str_patterns = get_patterns(relation_pattern_file)
    patterns = get_pattern_representation(str_patterns, model)
    scores = torch.mm(patterns, patterns.transpose(0,1))
    scores = scores * Variable(1.0 - torch.eye(scores.size(0)).cuda().float(), requires_grad=False)
    sorted_scores, indices = torch.topk(scores, 30, dim=1)
    sorted_scores, indices = sorted_scores.cpu().data.numpy(), indices.cpu().data.numpy()
    for i in range(indices.shape[0]):
        print(str_patterns[i], [(str_patterns[ind], score) for ind, score in zip(indices[i], sorted_scores[i])])
        print()


def get_model(model_dir):
    config_file = os.path.join(model_dir, 'saved_config.json')
    model_file = os.path.join(model_dir, 'best.pt')
    config = get_config(config_file)
    field = Field(batch_first=True, lower=True)
    create_vocab(config, field)
    arg_vocab = field.vocab
    rel_vocab = arg_vocab
    config.n_args = len(arg_vocab)
    model_type = getattr(config, 'model_type', 'sampling')
    if model_type == 'pairwise':
        model = PairwiseRelationalEmbeddingModel(config, arg_vocab, rel_vocab)
    elif model_type == 'sampling':
        model = RelationalEmbeddingModel(config, arg_vocab, rel_vocab)
    elif model_type == 'pair2seq':
        model = Pair2RelModel(config, arg_vocab, rel_vocab)
    else:
        raise NotImplementedError()
    load_model(model_file, model)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    model.cuda()
    return model

def get_topk_patterns(model, numpy_file):
    instances = np.load(numpy_file)
    instances = instances[:, 2:]
    unique, counts = np.unique(instances, return_counts=True, axis=0)
    count_sort_ind = np.argsort(-counts)
    sorted_unique, sorted_counts = unique[count_sort_ind], counts[count_sort_ind]
    topk = []
    for pattern, count in zip(sorted_unique[:100000], sorted_counts[:100000]):
        pattern_str = ' '.join([model.rel_vocab.itos[idx] for idx in pattern if idx != 1])
        print(pattern_str, count)
        topk += [pattern_str]
    with open('top100000.txt', 'w') as f:
        for p in topk:
            f.write('freq_based\t' + p + '\n')
    f.close()

def get_topk_args(model, numpy_file):
    instances = np.load(numpy_file)
    instances = instances[:, :2]
    unique, counts = np.unique(instances, return_counts=True, axis=0)
    count_sort_ind = np.argsort(-counts)
    sorted_unique, sorted_counts = unique[count_sort_ind], counts[count_sort_ind]
    topk = []
    for args, count in zip(sorted_unique[:100000], sorted_counts[:100000]):
        arg1, arg2 = model.arg_vocab.itos[args[0]], model.arg_vocab.itos[args[1]]
        topk += [arg1 + '\t' + arg2]
    with open('args_top100000.txt', 'w') as f:
        for p in topk:
            f.write('freq_based\t' + p + '\n')
    f.close()

if __name__ == '__main__':
    model_dir = sys.argv[1]
    model = get_model(model_dir)
    # numpy_file = sys.argv[2]
    # get_topk_patterns(model, numpy_file)
    # sys.exit(1)
    relation_instance_file = sys.argv[2]
    relation_pattern_file = sys.argv[3]
    neg_instance_file = sys.argv[4] if len(sys.argv) > 4 else None
    # out_dir = sys.argv[3]
    #word_pair_to_pairs(relation_instance_file, model)
    #pattern_to_patterns(relation_pattern_file, model)
    #word_pair_to_patterns(relation_pattern_file, relation_instance_file, model)
    patterns_to_kbc(relation_pattern_file, relation_instance_file, model, neg_instance_file)
