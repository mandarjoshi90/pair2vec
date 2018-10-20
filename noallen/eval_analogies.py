from collections import defaultdict
import numpy as np
from torch.autograd import Variable
from noallen.util import load_model, get_config
from noallen.model import RelationalEmbeddingModel
from noallen.torchtext.matrix_data import create_vocab
from noallen.torchtext.indexed_field import Field
import torch
import os
import sys
import fnmatch
from tqdm import tqdm
from random import shuffle
import random
from torch.nn.functional import softmax, normalize
from noallen.torchtext.vocab import Vectors

class DistributionalModel(torch.nn.Module):
    def __init__(self, vocab, dim, name='wikipedia-jan-18-model-300.vec', cache='/fasttext'):
        super(DistributionalModel, self).__init__()
        self.arg_vocab = vocab
        self.represent_arguments = torch.nn.Embedding(len(vocab), dim)
        self.represent_arguments.weight.requires_grad = False
        #self.arg_vocab.load_vectors(vectors='fasttext.en.300d', cache='/fasttext')
        #self.arg_vocab.load_vectors(vectors='glove.840B.300d', cache='/glove')
        #self.arg_vocab.vectors_cache = '~/data/fasttext'
        self.arg_vocab.load_vectors(Vectors(name=name, cache=cache))
        pretrained = self.arg_vocab.vectors
        # import ipdb
        # ipdb.set_trace()
        pretrained = normalize(pretrained) 
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

def read_g_analogies(fname, vocab):
    w1, w2, w3, w4 = [], [], [], []
    word1, word2, word3, word4 = [], [], [], []
    cat_to_tensors = {}
    cat = None
    oov, allw = 0, 0
    with open(fname, encoding='utf-8') as f:
        id_line = 0
        for id_line, line in enumerate(f):
            line = line.strip()
            try:
                if line.strip().startswith(':'):
                    if id_line > 0:
                        cat_to_tensors[cat] =  Variable(torch.LongTensor(w1), requires_grad=False).cuda(), Variable(torch.LongTensor(w2).cuda(), requires_grad=False).cuda(), Variable(torch.LongTensor(w3), requires_grad=False).cuda(), w4
                        w1, w2, w3, w4 = [], [], [], []
                    cat  = line
                    continue
                p1, p2, p3, p4 = line.lower().strip().split()
                w1.append(vocab.stoi[p1])
                w2.append(vocab.stoi[p2])
                w3.append(vocab.stoi[p3])
                w4.append([vocab.stoi[p4]])
                allw += 4
                oov += (vocab.stoi[p1] == 0) +  (vocab.stoi[p2] == 0) +  (vocab.stoi[p3] == 0) + (vocab.stoi[p4] == 0)
            except:
                print ("error reading pairs")
                print ("in file", fname)
                print ("in line",id_line,line)
                exit(-1)
    print('oov: ', oov * 100.0 / allw)
    return cat_to_tensors
    #return Variable(torch.LongTensor(w1), requires_grad=False).cuda(), Variable(torch.LongTensor(w2).cuda(), requires_grad=False).cuda(), Variable(torch.LongTensor(w3), requires_grad=False).cuda(), w4


def create_dataset(bats_dir):
    pairs = []
    for root, dirnames, filenames in os.walk(bats_dir):
        for filename in fnmatch.filter(sorted(filenames), '*'):
            pairs += read_pairs(os.path.join(root,filename))
    return pairs

def get_model(config_file, model_file):
    relemb_config = get_config(config_file, 'multiplication') 
    field = Field(batch_first=True)
    create_vocab(relemb_config, field)
    arg_vocab = field.vocab
    rel_vocab = arg_vocab
    relemb_config.n_args = len(arg_vocab)
    model = RelationalEmbeddingModel(relemb_config, arg_vocab, rel_vocab)
    load_model(model_file, model)
    return model

def get_relation_embedding(pair, model):
    word1, word2 = pair
    word1_embedding = model.represent_arguments(word1)
    word2_embedding = model.represent_arguments(word2)
    mlp_output = model.predict_relations(word1_embedding, word2_embedding)
    mlp_output = normalize(mlp_output, dim=-1)
    return mlp_output


def vocab_relation_embeddings(model, word1, direction):
    # (bs, dim)
    word1_embedding = model.represent_arguments(word1)
    # (V, dim)
    vocab_embedding = Variable(model.represent_arguments.weight.data, requires_grad=False)
    bs, dim = word1_embedding.size()
    vocab_size, _ = vocab_embedding.size()
    rep_word1_embedding = word1_embedding.unsqueeze(1).expand(bs, vocab_size, dim)
    rep_vocab_embedding = vocab_embedding.unsqueeze(0).expand(bs, vocab_size, dim)
    if direction == 'fwd':
        vocab_relemb_fwd = model.predict_relations(rep_word1_embedding.contiguous().view(-1, dim), rep_vocab_embedding.contiguous().view(-1, dim)).contiguous().view(bs, vocab_size, dim)
        vocab_relemb = normalize(vocab_relemb_fwd, dim=-1)
    elif direction == 'bwd':
        vocab_relemb_bwd = model.predict_relations(rep_vocab_embedding.contiguous().view(-1, dim), rep_word1_embedding.contiguous().view(-1, dim)).contiguous().view(bs, vocab_size, dim)
        vocab_relemb = normalize(vocab_relemb_bwd, dim=-1)
    else:
        vocab_relemb_fwd = model.predict_relations(rep_word1_embedding.contiguous().view(-1, dim), rep_vocab_embedding.contiguous().view(-1, dim)).contiguous().view(bs, vocab_size, dim)
        vocab_relemb_bwd = model.predict_relations(rep_vocab_embedding.contiguous().view(-1, dim), rep_word1_embedding.contiguous().view(-1, dim)).contiguous().view(bs, vocab_size, dim)
        vocab_relemb = normalize(vocab_relemb_fwd, dim=-1), normalize(vocab_relemb_bwd, dim=-1) 
    # return torch.cat((vocab_relemb_fwd, vocab_relemb_bwd), -1)
    return vocab_relemb

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

def get_mask(npy_dir, vocab, thr=150):
    if npy_dir is None:
        return None
    counts = np.zeros(len(vocab))
    for fname in tqdm(os.listdir(npy_dir)):
        if not fname.endswith('npy'):
            continue
        triples = np.load(os.path.join(npy_dir, fname))
        for i in range(0, triples.shape[0]):
            counts[triples[i, 0]] += 1
            counts[triples[i, 1]] += 1
    mask  = np.greater(counts, thr)
    mask = mask.astype(int)
    print(mask.shape, mask.sum())
    return mask

def get_bats_mask(bats_dir, vocab):
    mask = np.zeros(len(vocab))
    for root, dirnames, filenames in os.walk(bats_dir):
        for filename in fnmatch.filter(sorted(filenames), '*.txt'):
            pairs, idxs = read_pairs(os.path.join(root,filename), vocab)
            for left, right in idxs:
                mask[left] = 1
                for r in right:
                    mask[r] = 1
    print(mask.shape, mask.sum())
    return mask

def get_bats_mask_for_pairs(idxs, vocab):
    mask = np.zeros(len(vocab))
    for left, right in idxs:
        mask[left] = 1
        for r in right:
            mask[r] = 1
    print(mask.shape, mask.sum())
    return mask

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

def eval_on_google_combo(gfile, model_file, config_file, pred_file, batch=1, triplet_dir=None, direction='both'):
    random.seed(10)
    model = get_model(config_file, model_file)
    vocab = model.arg_vocab
    glove_model = DistributionalModel(vocab, 300)
    model.cuda()
    model.eval()
    glove_model.cuda()
    glove_model.eval()

    file_mask = np.ones(len(vocab))
    correct, total = 0, 0
    per_cat_acc, preds  = [], []
    cat_to_tensors = read_g_analogies(gfile, model.arg_vocab)
    for cat, tensors in cat_to_tensors.items():
        file_correct, file_total = 0, 0
        all_w1, all_w2, all_w3, all_w4 = tensors
        for i in tqdm(range(0, len(all_w1), batch)):
            w1, w2, w3, w4 = all_w1[i:i+batch], all_w2[i:i+batch], all_w3[i:i+batch], all_w4[i:i+batch]
            bs = len(w1)
            glove_scores = get_scores(glove_model, w1, w2, w3, direction, bs, method='3CosAdd')
            #scores = get_scores(model, w1, w2, w3, direction, bs, method='pair')
            scores =  glove_scores #+ scores
            mask = mask_out_analogy_words(file_mask, w1.data.cpu().numpy(), w2.data.cpu().numpy(), w3.data.cpu().numpy(), None)
            file_correct += get_accuracy(scores, w4, model.arg_vocab, w1, w2, w3, mask, i, preds, cat)
            file_total += len(w4)
        correct += file_correct
        total += file_total
        print(cat, file_correct * 100.0 / file_total, file_correct, file_total)
        print('cumulative', correct * 100.0 / total)
        per_cat_acc += [(cat,  file_correct, file_total)]
    print('Summary')
    group_correct = defaultdict(int)
    group_total = defaultdict(int)
    for cat, cat_correct, cat_total in per_cat_acc:
        group = 'Syn' if cat.startswith(': gram') else 'Sem'
        group_correct[group] += cat_correct
        group_total[group] += cat_total
        print(cat, cat_correct * 100.0 / cat_total)
    print('Final', correct * 100 / total)
    for group in group_correct.keys():
        acc = group_correct[group] * 100.0 / group_total[group]
        print(group, acc)

def get_scores(model, w1, w2, w3, direction, batch, method='3CosAdd'):
    vocab_size, dim = model.represent_arguments.weight.data.size()
    # (bs, dim)
    vocab_relemb = Variable(normalize(model.represent_arguments.weight.data, dim=-1), requires_grad=False).unsqueeze(0).expand(batch, vocab_size, dim)
    # (bs, V, dim))
    if method == '3CosAdd':
        p1_relemb = normalize(model.represent_arguments(w3) - model.represent_arguments(w1) + model.represent_arguments(w2), dim=-1)
        scores = torch.bmm(vocab_relemb, p1_relemb.unsqueeze(2)).squeeze(2)
    elif method == '3CosMul':
        cos_w4_w3 = (torch.bmm(vocab_relemb, normalize(model.represent_arguments(w3), dim=-1).unsqueeze(2)).squeeze(2) + 1) / 2
        cos_w4_w2 = (torch.bmm(vocab_relemb, normalize(model.represent_arguments(w2), dim=-1).unsqueeze(2)).squeeze(2) + 1) / 2
        cos_w4_w1 = (torch.bmm(vocab_relemb, normalize(model.represent_arguments(w1), dim=-1).unsqueeze(2)).squeeze(2) + 1) / 2
        scores = (cos_w4_w2 * cos_w4_w3) / (cos_w4_w1 + 0.00001)
    else:
        vocab_relemb = vocab_relation_embeddings(model, w3, direction)
        if direction == 'fwd':
            p1_relemb = get_relation_embedding((w1, w2), model)
            scores = (torch.bmm(vocab_relemb, p1_relemb.unsqueeze(2)).squeeze(2) + 1) / 2
        elif direction == 'bwd':
            p1_relemb = get_relation_embedding((w2, w1), model)
            scores = (torch.bmm(vocab_relemb, p1_relemb.unsqueeze(2)).squeeze(2) + 1) / 2
        else:
            p1_relemb_fwd, p1_relemb_bwd =  get_relation_embedding((w1, w2), model),  get_relation_embedding((w2, w1), model)
            vocab_relemb_fwd, vocab_relemb_bwd = vocab_relemb
            scores_fwd = (torch.bmm(vocab_relemb_fwd, p1_relemb_fwd.unsqueeze(2)).squeeze(2)) #+ 1) / 2
            scores_bwd = (torch.bmm(vocab_relemb_bwd, p1_relemb_bwd.unsqueeze(2)).squeeze(2)) #+ 1) / 2
            scores = (scores_fwd +  scores_bwd) / 2
    return scores

def eval_on_bats_combo(bats_dir, model_file, config_file, pred_file, batch=1, triplet_dir=None, direction='both'):
    random.seed(10)
    model = get_model(config_file, model_file)
    vocab = model.arg_vocab
    glove_model = DistributionalModel(vocab, 300)
    model.cuda()
    model.eval()
    glove_model.cuda()
    glove_model.eval()

    file_mask = np.ones(len(vocab))
    correct, total = 0, 0
    per_cat_acc, preds  = [], []
    for root, dirnames, filenames in os.walk(bats_dir):
        for filename in fnmatch.filter(sorted(filenames), '*.txt'):
            pairs, idxs = read_pairs(os.path.join(root,filename), model.arg_vocab)
            print(filename, len(idxs))
            # file_mask = get_bats_mask_for_pairs(idxs, model.arg_vocab)
            file_correct, file_total = 0, 0
            all_w1, all_w2, all_w3, all_w4 = pairs_to_analogies(idxs)

            bs = len(all_w1)
            print(bs)
            for i in tqdm(range(0, len(all_w1), batch)):
                w1, w2, w3, w4 = all_w1[i:i+batch], all_w2[i:i+batch], all_w3[i:i+batch], all_w4[i:i+batch]
                glove_scores = get_scores(glove_model, w1, w2, w3, direction, batch, method='3CosAdd')
                #scores = get_scores(model, w1, w2, w3, direction, batch, method='pair')
                scores =  glove_scores #+ scores
                mask = mask_out_analogy_words(file_mask, w1.data.cpu().numpy(), w2.data.cpu().numpy(), w3.data.cpu().numpy(), None)
                file_correct += get_accuracy(scores, w4, model.arg_vocab, w1, w2, w3, mask, i, preds, filename)
                file_total += len(w4)
            correct += file_correct
            total += file_total
            print(filename, file_correct * 100.0 / file_total, file_correct, file_total)
            print('cumulative', correct * 100.0 / total)
            per_cat_acc += [(filename,  file_correct,  file_total)]
    print('Summary')
    group_correct = defaultdict(int)
    group_total = defaultdict(int)
    for cat, cat_correct, cat_total in per_cat_acc:
        # group = 'Lexical' if cat[0] == 'L' else 'Enclyclopedic'
        group_correct[cat[0]] += cat_correct
        group_total[cat[0]] += cat_total
        print(cat, cat_correct * 100.0 / cat_total)
    print('Final', correct * 100 / total)
    for group in group_correct.keys():
        acc = group_correct[group] * 100.0 / group_total[group]
        print(group, acc)
    # with open(pred_file, encoding='utf-8', mode='w') as f:
        # for pred in preds:
            # f.write('\t'.join(pred) + '\n')

def eval_on_bats_interpolate(bats_dir, model_file, config_file, pred_file, batch=1, triplet_dir=None, direction='both'):
    random.seed(10)
    model = get_model(config_file, model_file)
    vocab = model.arg_vocab
    glove_model = DistributionalModel(vocab, 300)
    #model = DistributionalModel(vocab, 300, name='glove.840B.300d', cache='/glove')
    model.cuda()
    model.eval()
    glove_model.cuda()
    glove_model.eval()

    file_mask = np.ones(len(vocab))
    correct, total = 0, 0
    per_cat_acc, preds  = [], []
    all_alpha_acc = []
    for root, dirnames, filenames in os.walk(bats_dir):
        for filename in fnmatch.filter(sorted(filenames), '*.txt'):
            pairs, idxs = read_pairs(os.path.join(root,filename), model.arg_vocab)
            print(filename, len(idxs))
            # file_mask = get_bats_mask_for_pairs(idxs, model.arg_vocab)
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
                    glove_scores = get_scores(glove_model, w1, w2, w3, direction, batch, method='3CosAdd')
                    scores = get_scores(model, w1, w2, w3, direction, batch, method='pairwise')
                    scores =  alpha * glove_scores + (1- alpha) * scores
                    mask = mask_out_analogy_words(file_mask, w1.data.cpu().numpy(), w2.data.cpu().numpy(), w3.data.cpu().numpy(), None)
                    file_correct += get_accuracy(scores, w4, model.arg_vocab, w1, w2, w3, mask, i, preds, filename)
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
        # group = 'Lexical' if cat[0] == 'L' else 'Enclyclopedic'
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
# def eval_on_bats(bats_dir, model_file, config_file, pred_file, batch=2, triplet_dir=None, direction='fwd'):
    # random.seed(10)
    # model = get_model(config_file, model_file)
    # vocab = model.arg_vocab
    # if True:
        # model = DistributionalModel(vocab, 300)
        # print("-------------Disributed-----------------")
    # model.cuda()
    # model.eval()

    # print('--------------', direction, '--------------')
    # # mask = get_mask(triplet_dir, model.arg_vocab)
    # # file_mask = get_bats_mask(bats_dir, model.arg_vocab)
    # file_mask = np.ones(len(vocab))
    # correct, total = 0, 0
    # per_cat_acc, preds  = [], []
    # for root, dirnames, filenames in os.walk(bats_dir):
        # for filename in fnmatch.filter(sorted(filenames), '[LE]*.txt'):
            # pairs, idxs = read_pairs(os.path.join(root,filename), model.arg_vocab)
            # print(filename, len(idxs))
            # # file_mask = get_bats_mask_for_pairs(idxs, model.arg_vocab)
            # file_correct, file_total = 0, 0
            # all_w1, all_w2, all_w3, all_w4 = pairs_to_analogies(idxs)

            # bs = len(all_w1)
            # print(bs)
            # for i in tqdm(range(0, len(all_w1), batch)):
                # w1, w2, w3, w4 = all_w1[i:i+batch], all_w2[i:i+batch], all_w3[i:i+batch], all_w4[i:i+batch]
                # vocab_size, dim = model.represent_arguments.weight.data.size()
                # # (bs, dim)
                # if direction == 'fwd':
                    # p1_relemb = get_relation_embedding((w1, w2), model)
                # elif direction == 'bwd':
                    # p1_relemb = get_relation_embedding((w2, w1), model)
                # else:
                    # p1_relemb_fwd, p1_relemb_bwd = get_relation_embedding((w1, w2), model), get_relation_embedding((w2, w1), model)
                    # p1_relemb = torch.cat((p1_relemb_fwd, p1_relemb_bwd), -1)
                # # (bs, V, dim))
                # #p1_relemb = normalize(model.represent_arguments(w3) - model.represent_arguments(w1) + model.represent_arguments(w2))
                # #vocab_relemb = Variable(normalize(model.represent_arguments.weight.data), requires_grad=False).unsqueeze(0).expand(batch, vocab_size, dim)
                # vocab_relemb = vocab_relation_embeddings(model, w3, direction)
                # scores = torch.bmm(vocab_relemb, p1_relemb.unsqueeze(2)).squeeze(2)
                # # scores = (pair1_rel_emb.unsqueeze(1).expand(bs, vocab_size, dim) * vocab_rel_emb).sum(-1)
                # mask = mask_out_analogy_words(file_mask, w1.data.cpu().numpy(), w2.data.cpu().numpy(), w3.data.cpu().numpy(), model)
                # file_correct += get_accuracy(scores, w4, model.arg_vocab, w1, w2, w3, mask, i, preds, filename)
                # file_total += len(w4)
            # correct += file_correct
            # total += file_total
            # print(filename, file_correct * 100.0 / file_total, file_correct, file_total)
            # print('cumulative', correct * 100.0 / total)
            # per_cat_acc += [(filename,  file_correct * 100.0 / file_total)]
    # print('Summary')
    # for cat, acc in per_cat_acc:
        # print(cat, acc)
    # print('Final', correct * 100 / total)
    # # with open(pred_file, encoding='utf-8', mode='w') as f:
        # # for pred in preds:
            # # f.write('\t'.join(pred) + '\n')

bats_dir = sys.argv[1]
model_dir = sys.argv[2]
output_file = sys.argv[3]
triplet_dir = sys.argv[4] if len(sys.argv) == 5 else None
eval_on_bats_interpolate(bats_dir, os.path.join(model_dir, 'best.pt'), os.path.join(model_dir, 'relemb.json'), output_file, triplet_dir=triplet_dir)
