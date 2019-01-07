from collections import defaultdict
import numpy as np
from torch.autograd import Variable
from embeddings.model import Pair2Vec
from embeddings.matrix_data import create_vocab
from embeddings.indexed_field import Field
from endtasks.util import get_pair2vec
import torch
import os
import sys
import fnmatch
from tqdm import tqdm
from random import shuffle
import random
from torch.nn.functional import softmax, normalize
from embeddings.vocab import Vectors

class DistributionalModel(torch.nn.Module):
    def __init__(self, vocab, dim, name='wikipedia-jan-18-model-300.vec', cache='/fasttext'):
        super(DistributionalModel, self).__init__()
        self.arg_vocab = vocab
        self.represent_arguments = torch.nn.Embedding(len(vocab), dim)
        self.represent_arguments.weight.requires_grad = False
        self.arg_vocab.load_vectors(Vectors(name=name, cache=cache))
        pretrained = self.arg_vocab.vectors
        #pretrained = normalize(pretrained) 
        self.represent_arguments.weight.data.copy_(pretrained)

    def forward(self):
        pass


    def predict_relations(self, subjects, objects):
        return subjects - objects

def read_pairs(fname, vocab):
    pairs, idxs = [], []
    oov, total = 0, 0
    with open(fname, encoding='utf-8') as f:
        id_line = 0
        for id_line, line in enumerate(f):
            try:
                if "\t" in line:
                    left,right = line.lower().split("\t")
                else:
                    left,right = line.lower().split()
                right = right.strip()
                if "/" in right:
                    right=[i.strip() for i in right.split("/")]
                else:
                    right=[i.strip() for i in right.split(",")]
                left_idx = vocab.stoi[left]
                right = [r for r in right if vocab.stoi[r] != 0]
                right_idxs = [vocab.stoi[r] for r in right]
                total += 1 + len(right_idxs)
                oov += int(left_idx == 0) + sum(int(r == 0) for r in right_idxs)
                if left_idx != 0 and len(right_idxs) > 0:
                    pairs.append([left,right])
                    idxs.append([left_idx, right_idxs])
            except:
                print ("error reading pairs")
                print ("in file", fname)
                print ("in line",id_line,line)
                exit(-1)
    print('oov', oov * 100.0 / total)
    return pairs, idxs



def create_dataset(bats_dir):
    pairs = []
    for root, dirnames, filenames in os.walk(bats_dir):
        for filename in fnmatch.filter(sorted(filenames), '*'):
            pairs += read_pairs(os.path.join(root,filename))
    return pairs


def predict_relations(pair, model):
    word1, word2 = pair
    word1_embedding = model.represent_arguments(word1)
    word2_embedding = model.represent_arguments(word2)
    mlp_output = model.predict_relations(word1_embedding, word2_embedding)
    mlp_output = normalize(mlp_output, dim=-1)
    return mlp_output


def vocab_pair_embeddings(model, word1):
    # (bs, dim)
    word1_embedding = model.represent_arguments(word1)
    # (V, dim)
    vocab_embedding = Variable(model.represent_arguments.weight.data, requires_grad=False)
    bs, dim = word1_embedding.size()
    vocab_size, _ = vocab_embedding.size()
    rep_word1_embedding = word1_embedding.unsqueeze(1).expand(bs, vocab_size, dim)
    rep_vocab_embedding = vocab_embedding.unsqueeze(0).expand(bs, vocab_size, dim)
    vocab_pair_fwd = model.predict_relations(rep_word1_embedding.contiguous().view(-1, dim), rep_vocab_embedding.contiguous().view(-1, dim)).contiguous().view(bs, vocab_size, dim)
    vocab_pair_bwd = model.predict_relations(rep_vocab_embedding.contiguous().view(-1, dim), rep_word1_embedding.contiguous().view(-1, dim)).contiguous().view(bs, vocab_size, dim)
    vocab_pair = normalize(vocab_pair_fwd, dim=-1), normalize(vocab_pair_bwd, dim=-1) 
    return vocab_pair

def pairs_to_analogies(pairs):
    tups = []
    for pair1 in pairs:
        for pair2 in pairs:
            if pair1 != pair2:
                tups.append((pair1[0], pair1[1][0], pair2[0], pair2[1]))
    shuffle(tups)
    get = lambda i : [x[i] for x in tups]
    w1, w2, w3, w4 = get(0), get(1), get(2), get(3)
    return Variable(torch.LongTensor(w1), requires_grad=False).cuda(), Variable(torch.LongTensor(w2).cuda(), requires_grad=False).cuda(), Variable(torch.LongTensor(w3), requires_grad=False).cuda(), w4


def get_accuracy(org_scores, w4, vocab, w1, w2, w3, mask, batch_num, preds, filename):
    if mask is not None:
        mask = Variable(torch.from_numpy(mask).cuda(), requires_grad=False).float()
        scores = (org_scores - (org_scores.min(-1, keepdim=True)[0])) * mask
    else:
        scores = org_scores
    sorted_scores, indices = torch.sort(scores, descending=True, dim=-1)
    w1, w2, w3 = w1.data.cpu().numpy().tolist(), w2.data.cpu().numpy().tolist(), w3.data.cpu().numpy().tolist()
    predictions = indices[:, 0].cpu().data.numpy().tolist()
    acc = 0
    for i, (pred, gold) in enumerate(zip(predictions, w4)):
        ranks =  indices[i].cpu().data.numpy().tolist()
        topk =  indices[i, :10].cpu().data.numpy().tolist()
        topk_scores =  sorted_scores[i, :10].cpu().data.numpy().tolist()
        topk = [vocab.itos[w] for w in topk]

        gold_ranks = [ranks.index(g) for g in gold]
        preds += [(filename, vocab.itos[w1[i]], vocab.itos[w2[i]], vocab.itos[w3[i]], '\t'.join(topk), '\t'.join([vocab.itos[g] for g in gold]))]
        if pred in gold:
            acc += 1
        # if batch_num < 15:
            # print(vocab.itos[w1[i]], ':', vocab.itos[w2[i]], '::', vocab.itos[w3[i]], ':', vocab.itos[gold[0]], min(gold_ranks), topk)
        topk =  indices[i, :10].cpu().data.numpy().tolist()

    return acc

def mask_out_analogy_words(file_mask, w1_batch, w2_batch, w3_batch, model):
    mask = np.tile(file_mask.copy(), (w1_batch.shape[0], 1))
    for i, (w1, w2, w3) in enumerate(zip(w1_batch, w2_batch, w3_batch)):
        mask[i, w1] = 0
        mask[i, w2] = 0
        mask[i, w3] = 0
    return mask


def get_scores(model, w1, w2, w3, batch, method='3CosAdd'):
    vocab_size, dim = model.represent_arguments.weight.data.size()
    # (bs, V, dim))
    if method == '3CosAdd':
        vocab_emb = Variable(normalize(model.represent_arguments.weight.data, dim=-1), requires_grad=False).unsqueeze(0).expand(batch, vocab_size, dim)
        p1_relemb = normalize(model.represent_arguments(w3) - model.represent_arguments(w1) + model.represent_arguments(w2), dim=-1)
        scores = torch.bmm(vocab_emb, p1_relemb.unsqueeze(2)).squeeze(2)
    else:
        vocab_pair = vocab_pair_embeddings(model, w3)
        p1_fwd, p1_bwd =  predict_relations((w1, w2), model),  predict_relations((w2, w1), model)
        vocab_pair_fwd, vocab_pair_bwd = vocab_pair
        scores_fwd = (torch.bmm(vocab_pair_fwd, p1_fwd.unsqueeze(2)).squeeze(2))
        scores_bwd = (torch.bmm(vocab_pair_bwd, p1_bwd.unsqueeze(2)).squeeze(2))
        scores = (scores_fwd +  scores_bwd) / 2
    return scores

def eval_on_bats_interpolate(bats_dir, model_file, config_file, pred_file, batch=1):
    random.seed(10)
    pair2vec = get_pair2vec(config_file, model_file)
    vocab = pair2vec.arg_vocab
    distrib_model = DistributionalModel(vocab, 300)
    pair2vec.cuda()
    pair2vec.eval()
    distrib_model.cuda()
    distrib_model.eval()

    file_mask = np.ones(len(vocab))
    correct, total = 0, 0
    per_cat_acc, preds  = [], []
    all_alpha_acc = []
    for root, dirnames, filenames in os.walk(bats_dir):
        for filename in fnmatch.filter(sorted(filenames), '*.txt'):
            pairs, idxs = read_pairs(os.path.join(root,filename), pair2vec.arg_vocab)
            print(filename, len(idxs))
            best_correct, best_alpha = 0, 0
            for alpha in np.linspace(0,1,11):
                alpha = float(alpha)
                print('alpha', alpha)
                file_correct, file_total = 0, 0
                all_w1, all_w2, all_w3, all_w4 = pairs_to_analogies(idxs)

                bs = len(all_w1)
                print(bs)
                for i in tqdm(range(0, len(all_w1), batch)):
                    w1, w2, w3, w4 = all_w1[i:i+batch], all_w2[i:i+batch], all_w3[i:i+batch], all_w4[i:i+batch]
                    distrib_scores = get_scores(distrib_model, w1, w2, w3, batch, method='3CosAdd')
                    scores = get_scores(pair2vec, w1, w2, w3, batch, method='pair2vec')
                    scores =  alpha * distrib_scores + (1- alpha) * scores
                    mask = mask_out_analogy_words(file_mask, w1.data.cpu().numpy(), w2.data.cpu().numpy(), w3.data.cpu().numpy(), None)
                    file_correct += get_accuracy(scores, w4, pair2vec.arg_vocab, w1, w2, w3, mask, i, preds, filename)
                    file_total += len(w4)
                print(filename, file_correct * 100.0 / file_total, file_correct, file_total, alpha)
                all_alpha_acc.append((filename, file_correct, file_total, file_correct * 100.0 / file_total, alpha))
                if file_correct > best_correct:
                    best_correct, best_alpha = file_correct, alpha
            file_correct = best_correct
            correct += file_correct
            total += file_total
            print(filename, file_correct * 100.0 / file_total, file_correct, file_total, best_alpha)
            print('cumulative', correct * 100.0 / total)
            per_cat_acc += [(filename,  file_correct,  file_total, best_alpha)]
    print('Summary')
    group_correct = defaultdict(int)
    group_total = defaultdict(int)
    for cat, cat_correct, cat_total, best_alpha in per_cat_acc:
        group_correct[cat[0]] += cat_correct
        group_total[cat[0]] += cat_total
        print(cat, cat_correct * 100.0 / cat_total, best_alpha)
    print('Final', correct * 100 / total)
    for group in group_correct.keys():
        acc = group_correct[group] * 100.0 / group_total[group]
        print(group, acc)
    with open(pred_file, encoding='utf-8', mode='w') as f:
        for info in all_alpha_acc:
            f.write('\t'.join([str(x) for x in info]) + '\n')

if __name__ == '__main__':
    bats_dir = sys.argv[1]
    model_dir = sys.argv[2]
    output_file = sys.argv[3]
    eval_on_bats_interpolate(bats_dir, os.path.join(model_dir, 'best.pt'), os.path.join(model_dir, 'saved_config.json'), output_file)
