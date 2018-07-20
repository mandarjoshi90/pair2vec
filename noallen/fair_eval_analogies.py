import numpy as np
from noallen.torchtext.vocab import Vocab
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

class DistributionalModel(torch.nn.Module):
    def __init__(self, vocab, dim):
        super(DistributionalModel, self).__init__()
        self.arg_vocab = vocab
        self.represent_arguments = torch.nn.Embedding(len(vocab), dim)
        self.represent_arguments.weight.requires_grad = False
        self.arg_vocab.load_vectors('glove.6B.300d')
        pretrained = self.arg_vocab.vectors
        pretrained = normalize(pretrained) 
        self.represent_arguments.weight.data.copy_(pretrained)

    def forward(self):
        pass

    # def represent_arguments(subject):
        # return self.embedding(subject)

    def predict_relations(self, subjects, objects):
        return subjects - objects

def read_pairs(fname):
    pairs = []
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
                pairs.append((left, right))
            except:
                print ("error reading pairs")
                print ("in file", fname)
                print ("in line",id_line,line)
                exit(-1)
    return pairs

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
    #if isinstance(model, RelationalEmbeddingModel):
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
        vocab_relemb = normalize(vocab_relemb_fwd, dim=2)
    elif direction == 'bwd':
        vocab_relemb_bwd = model.predict_relations(rep_vocab_embedding.contiguous().view(-1, dim), rep_word1_embedding.contiguous().view(-1, dim)).contiguous().view(bs, vocab_size, dim)
        vocab_relemb = normalize(vocab_relemb_bwd, dim=2)
    else:
        vocab_relemb_fwd = model.predict_relations(rep_word1_embedding.contiguous().view(-1, dim), rep_vocab_embedding.contiguous().view(-1, dim)).contiguous().view(bs, vocab_size, dim)
        vocab_relemb_bwd = model.predict_relations(rep_vocab_embedding.contiguous().view(-1, dim), rep_word1_embedding.contiguous().view(-1, dim)).contiguous().view(bs, vocab_size, dim)
        vocab_relemb = torch.cat((normalize(vocab_relemb_fwd, dim=2),normalize(vocab_relemb_bwd, dim=2) ), -1)
    # return torch.cat((vocab_relemb_fwd, vocab_relemb_bwd), -1)
    return vocab_relemb

def pairs_to_analogies(pairs, vocab):
    tups = []
    for pair1 in pairs:
        for pair2 in pairs:
            if pair1 != pair2:
                tups.append((vocab.stoi[pair1[0]], vocab.stoi[pair1[1][0]], vocab.stoi[pair2[0]], pair2[1], (pair1, pair2)))
    shuffle(tups)
    get = lambda i : [x[i] for x in tups]
    w1, w2, w3, w4, quad = get(0), get(1), get(2), get(3), get(4)
    return Variable(torch.LongTensor(w1), requires_grad=False).cuda(), Variable(torch.LongTensor(w2).cuda(), requires_grad=False).cuda(), Variable(torch.LongTensor(w3), requires_grad=False).cuda(), w4, quad

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
    # import ipdb
    # ipdb.set_trace()
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

def get_accuracy(vocab, org_scores, w1, w2, w3, w4, batch_num, preds, filename):
    sorted_scores, indices = torch.sort(org_scores, descending=True, dim=-1)
    w1, w2, w3 = w1.data.cpu().numpy().tolist(), w2.data.cpu().numpy().tolist(), w3.data.cpu().numpy().tolist()
    # predictions = indices[:, 0].cpu().data.numpy().tolist()
    org_scores = org_scores.data.cpu().numpy()
    acc = 0
    for i, (gold) in enumerate(w4):
        exclude = set([w1[i], w2[i], w3[i]])
        ranks = indices[i].cpu().data.numpy().tolist()
        topk = indices[i, :10].cpu().data.numpy().tolist()
        topk = [vocab.itos[w] for w in topk if w not in exclude]
        # import ipdb
        # ipdb.set_trace()
        gold_ranks = [ranks.index(vocab.stoi[g]) for g in gold]
        pred = topk[0]
        # preds += [(filename, vocab.itos[w1[i]], vocab.itos[w2[i]], vocab.itos[w3[i]], '\t'.join(topk), '\t'.join([vocab.itos[g] for g in gold]))]
        if pred in gold:
            acc += 1
        # else:
            # print(vocab.itos[w1[i]], ':', vocab.itos[w2[i]], '::', vocab.itos[w3[i]], ':', gold[0], min(gold_ranks), topk)
            # if batch_num < 150:
                # print(vocab.itos[w1[i]], ':', vocab.itos[w2[i]], '::', vocab.itos[w3[i]], ':', vocab.itos[gold[0]], '/', vocab.itos[pred], min(gold_ranks), topk)
        if batch_num < 15:
            # import ipdb
            # ipdb.set_trace()
            topk = [(vocab.itos[w], org_scores[i, w]) for w in ranks[:10] if w not in exclude]
            min_index, min_rank = min(enumerate(gold_ranks), key=lambda p : p[1])
            min_rank_score = org_scores[i, vocab.stoi[gold[min_index]]]
            print(vocab.itos[w1[i]], ':', vocab.itos[w2[i]], '::', vocab.itos[w3[i]], ':', gold[0], min_rank, min_rank_score, topk)

    return acc

def mask_out_analogy_words(file_mask, w1_batch, w2_batch, w3_batch, model):
    mask = np.tile(file_mask.copy(), (w1_batch.shape[0], 1))
    for i, (w1, w2, w3) in enumerate(zip(w1_batch, w2_batch, w3_batch)):
        mask[i, w1] = 0
        mask[i, w2] = 0
        mask[i, w3] = 0
    return mask

def read_vocab_file(fname):
    vocab = []
    with open (fname, encoding='utf-8') as f:
        for line in f:
            vocab.append(line.strip())
    return vocab

def distributional_topk_mask(dmodel, w3, k=50):
    w3_embed = dmodel.represent_arguments(w3)
    batch, dim = w3_embed.size()
    vocab_size, _ = dmodel.represent_arguments.weight.data.size()

    vocab_embed = Variable((dmodel.represent_arguments.weight.data), requires_grad=False).unsqueeze(0).expand(batch, vocab_size, dim)
    scores = torch.bmm(vocab_embed, w3_embed.unsqueeze(2)).squeeze(2)
    _, topk_indices = torch.topk(scores, k, dim=-1)
    mask = torch.zeros_like(scores)
    mask.scatter_(1, topk_indices, 1)
    return mask
    


def eval_distributional_on_bats(bats_dir, vocab_file, pred_file, batch=2, triplet_dir=None, direction='fwd'):
    print("-------------Disributed-----------------")
    random.seed(10)
    vocab = Vocab(read_vocab_file(vocab_file), specials=['<unk>'])
    model = DistributionalModel(vocab, 300)
    model.cuda()
    model.eval()
    correct, total = 0, 0
    per_cat_acc, preds  = [], []
    for root, dirnames, filenames in os.walk(bats_dir):
        for filename in fnmatch.filter(sorted(filenames), '[LE]*.txt'):
            pairs = read_pairs(os.path.join(root,filename))
            print(filename, len(pairs))
            file_correct, file_total = 0, 0
            all_w1, all_w2, all_w3, all_w4, quads = pairs_to_analogies(pairs, vocab)
            for i in tqdm(range(0, len(all_w1), batch)):
                w1, w2, w3, w4 = all_w1[i:i+batch], all_w2[i:i+batch], all_w3[i:i+batch], all_w4[i:i+batch]
                vocab_size, dim = model.represent_arguments.weight.data.size()
                # (bs, dim)
                # p1_relemb = get_relation_embedding((w1, w2), model)
                # vocab_relemb = vocab_relation_embeddings(model, w3, direction)
                p1_relemb = normalize(model.represent_arguments(w3) - model.represent_arguments(w1) + model.represent_arguments(w2))
                vocab_relemb = Variable(normalize(model.represent_arguments.weight.data), requires_grad=False).unsqueeze(0).expand(batch, vocab_size, dim)
                # import ipdb
                # ipdb.set_trace()
                # (bs, V, dim))
                scores = torch.bmm(vocab_relemb, p1_relemb.unsqueeze(2)).squeeze(2)
                mask = distributional_topk_mask(model, w3)
                scores.masked_fill_((1 - mask).byte(), -1e20)
                file_correct += get_accuracy(vocab, scores, w1, w2, w3, w4, i, preds, filename)
                file_total += len(w4)
            correct += file_correct
            total += file_total
            print(filename, file_correct * 100.0 / file_total, file_correct, file_total)
            print('cumulative', correct * 100.0 / total)
            per_cat_acc += [(filename,  file_correct * 100.0 / file_total)]
    print('Summary')
    for cat, acc in per_cat_acc:
        print(cat, acc)
    print('Final', correct * 100 / total)

def eval_on_bats(bats_dir, model_file, config_file, pred_file, batch=2, triplet_dir=None, direction='fwd'):
    random.seed(10)
    model = get_model(config_file, model_file)
    vocab = model.arg_vocab
    dmodel = DistributionalModel(vocab, 300)
    dmodel.cuda()
    dmodel.eval()
    model.cuda()
    model.eval()
    print('--------------', direction, '--------------')
    # mask = get_mask(triplet_dir, model.arg_vocab)
    file_mask = np.ones(len(model.arg_vocab)) #get_bats_mask(bats_dir, model.arg_vocab)
    file_mask[0:4] = 0
    correct, total = 0, 0
    per_cat_acc, preds  = [], []
    for root, dirnames, filenames in os.walk(bats_dir):
        for filename in fnmatch.filter(sorted(filenames), '[LE]*.txt'):
            pairs  = read_pairs(os.path.join(root,filename))
            print(filename, len(pairs))
            #file_mask = get_bats_mask_for_pairs(idxs, model.arg_vocab)
            file_correct, file_total = 0, 0
            all_w1, all_w2, all_w3, all_w4, quads = pairs_to_analogies(pairs, vocab)

            bs = len(all_w1)
            for i in tqdm(range(0, len(all_w1), batch)):
                w1, w2, w3, w4 = all_w1[i:i+batch], all_w2[i:i+batch], all_w3[i:i+batch], all_w4[i:i+batch]
                vocab_size, dim = model.represent_arguments.weight.data.size()
                # (bs, dim)
                if direction == 'fwd':
                    p1_relemb = get_relation_embedding((w1, w2), model)
                elif direction == 'bwd':
                    p1_relemb = get_relation_embedding((w2, w1), model)
                else:
                    p1_relemb_fwd, p1_relemb_bwd = get_relation_embedding((w1, w2), model), get_relation_embedding((w2, w1), model)
                    p1_relemb = torch.cat((p1_relemb_fwd, p1_relemb_bwd), -1)
                # (bs, V, dim))
                vocab_relemb = vocab_relation_embeddings(model, w3, direction)
                scores = torch.bmm(vocab_relemb, p1_relemb.unsqueeze(2)).squeeze(2)
                mask = distributional_topk_mask(dmodel, w3, k=500)
                scores.masked_fill_((1 - mask).byte(), -1e20)
                # scores = (pair1_rel_emb.unsqueeze(1).expand(bs, vocab_size, dim) * vocab_rel_emb).sum(-1)
                file_correct += get_accuracy(vocab, scores, w1, w2, w3, w4, i, preds, filename)
                file_total += len(w4)
            correct += file_correct
            total += file_total
            print(filename, file_correct * 100.0 / file_total, file_correct, file_total)
            print('cumulative', correct * 100.0 / total)
            per_cat_acc += [(filename,  file_correct * 100.0 / file_total)]
    print('Summary')
    for cat, acc in per_cat_acc:
        print(cat, acc)
    print('Final', correct * 100 / total)
    with open(pred_file, encoding='utf-8', mode='w') as f:
        for pred in preds:
            f.write('\t'.join(pred) + '\n')

bats_dir = sys.argv[1]
if len(sys.argv) == 3:
    vocab_file = sys.argv[2]
    eval_distributional_on_bats(bats_dir, vocab_file, None)
else:
    model_dir = sys.argv[2]
    output_file = sys.argv[3]
    triplet_dir = sys.argv[4] if len(sys.argv) == 5 else None
    eval_on_bats(bats_dir, os.path.join(model_dir, 'best.pt'), os.path.join(model_dir, 'relemb.json'), output_file, triplet_dir=triplet_dir)
