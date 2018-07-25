from noallen.torchtext.vocab import Vocab
from tqdm import tqdm
import torch
from torch.autograd import Variable
from torch.nn import Embedding
from noallen.torchtext.indexed_field import Field
from torch.nn.functional import cosine_similarity
import numpy as np
from torch.nn.functional import softmax, normalize

def get_pairs(words):
    pairs = [(i,j) for i in range(len(words)) for j in range(len(words))]
    w1, w2 = [w[0] for w in pairs], [w[1] for w in pairs]
    return w1, w2

def get_words(filename):
    with open(filename, encoding='utf-8') as f:
        text = f.read().lower().strip().split('\n')
        return text

def dump_pair_sims(start, end, words, scores, filename, mode):
    pair_scores = [(words[i], words[j], str(round(scores[i-start,j],2))) for i in range(start,end) for j in range(len(words)) if scores[i-start,j] > 0.2]
    with open(filename, mode=mode, encoding='utf-8') as f:
        for trip in pair_scores:
            f.write('\t'.join(trip) + '\n')

def get_embedding(vocab):
    embedding = Embedding(len(vocab), 300)
    embedding.requires_grad = False
    pretrained = normalize(vocab.vectors)
    embedding.weight.data.copy_(pretrained)
    embedding.eval()
    return embedding

def cossim(w1_embedding, w2_embedding, eps=1e-8):
    (bs, d), (v, d) = w1_embedding.size(), w2_embedding.size()
    scores = torch.mm(w1_embedding, w2_embedding.permute(1,0))
    w1 = torch.norm(w1_embedding, 2, -1).unsqueeze(1).expand(bs, v)
    w2 = torch.norm(w2_embedding, 2, -1).unsqueeze(0).expand(bs,v)
    return scores / (w1 * w2).clamp(min=eps)

def dot(w1_embedding, w2_embedding, eps=1e-8):
    (bs, d), (v, d) = w1_embedding.size(), w2_embedding.size()
    scores = torch.mm(w1_embedding, w2_embedding.permute(1,0))
    return scores 

def pairsims(vocab_file, out_file, bs=100):
    words = get_words(vocab_file)
    print('done reading vocab')
    field = Field(batch_first=True, sequential=False)
    #vocab = Vocab(words,specials=[], vectors='fasttext.en.300d', vectors_cache='/fasttext')
    vocab = Vocab(words,specials=[], vectors='glove.6B.300d', vectors_cache='/glove')
    field.vocab = vocab
    embedding  = get_embedding(vocab)
    embedding.cuda()
    mode = 'w'
    print('start')
    for start in tqdm(range(0, len(words), bs)):
        word1 = field.process(words[start:start+bs], device=None, train=False)
        # word1, word2  = field.process(pairs[0][start:start+bs], device=None, train=False), field.process(pairs[1][start:start+bs], device=None, train=False)
        w1_embedding = embedding(word1)
        w2_embedding = embedding.weight.data
        bs, _ = w1_embedding.size()
        scores  = cossim(w1_embedding, w2_embedding)
        scores_list = scores.data.cpu().numpy()
        #pair_scores += [(w1, w2, str(round(s,2))) for w1,w2,s in zip(pairs[0][start:start+bs], pairs[1][start:start+bs], scores_list)]
        dump_pair_sims(start,start+bs, words, scores_list, out_file, mode)
        mode = 'a'

def print_topk(word, vocab, topk_indxs):
    word_i = vocab.stoi[word]
    topk = topk_indxs[word_i]
    topk_words = [vocab.itos[i] for i in topk]
    return topk_words

def topk(vocab_file, topk_scores_file, topk_indxs_file, bs=100, k=100):
    words = get_words(vocab_file)
    print('done reading vocab')
    field = Field(batch_first=True, sequential=False)
    #vocab = Vocab(words,specials=[], vectors='fasttext.en.300d', vectors_cache='/fasttext')
    vocab = Vocab(words,specials=[], vectors='glove.6B.300d', vectors_cache='/glove')
    field.vocab = vocab
    embedding  = get_embedding(vocab)
    embedding.cuda()
    mode = 'w'
    topk_scores_list, topk_indx_list = [], []
    zero_diag = Variable(1.0 - torch.eye(bs, len(vocab)), requires_grad=False).cuda()

    print('start')
    for start in tqdm(range(0, len(words), bs)):
        if start + bs > len(words):
            bs = len(words) - start
        eye = torch.eye(bs, len(vocab)) if start == 0 else torch.cat((torch.zeros(bs, start), torch.eye(bs, bs)), dim=-1)
        if len(vocab) - start - bs > 0 and start != 0:
            eye = torch.cat((eye, torch.zeros(bs, len(vocab) - start - bs)), dim=-1)
        eye = eye.cuda()
        word1 = field.process(words[start:start+bs], device=None, train=False)
        # word1, word2  = field.process(pairs[0][start:start+bs], device=None, train=False), field.process(pairs[1][start:start+bs], device=None, train=False)
        w1_embedding = embedding(word1)
        w2_embedding = embedding.weight
        zero_diag = Variable(1.0 - eye, requires_grad=False)
        # scores  = cossim(w1_embedding, w2_embedding) * zero_diag
        scores = torch.mm(w1_embedding, w2_embedding.permute(1,0))* zero_diag
        topk_scores, topk_indxs = torch.topk(scores, k)
        # topk_scores = softmax(topk_scores, dim=-1) 
        topk_scores = (topk_scores + 1) / (topk_scores + 1).sum(-1, keepdim=True)
        # import ipdb
        # ipdb.set_trace()
        topk_scores_list.append(topk_scores.data.cpu().numpy())
        topk_indx_list.append(topk_indxs.data.cpu().numpy())
        # break
    topk_scores, topk_indxs = np.concatenate(topk_scores_list), np.concatenate(topk_indx_list)
    np.save(topk_scores_file, topk_scores)
    np.save(topk_indxs_file, topk_indxs)
vocab_file = '/sdb/data/wikipedia-sentences/expanded_softsample/vocab.txt'
# out_file = '/sdb/data/wikipedia-sentences/pairs.txt'
topk_scores_file = '/sdb/data/wikipedia-sentences/expanded_softsample/topk_scores.npy'
topk_indxs_file = '/sdb/data/wikipedia-sentences/expanded_softsample/topk_indxs.npy'
topk(vocab_file, topk_scores_file, topk_indxs_file)
