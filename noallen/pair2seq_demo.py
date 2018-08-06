import sys
import torch
from torch.autograd import Variable
from noallen.model import Pair2RelModel
from noallen.torchtext.vocab import Vocab
from noallen.torchtext.indexed_field import Field
from noallen.util import load_model, get_args, get_config
import os
from torch.nn.functional import log_softmax

def get_model(config_file, model_file):
    config = get_config(config_file)
    args_field = Field(lower=True, batch_first=True, sequential=config.compositional_args)
    vocab_path = os.path.join(config.triplet_dir, 'vocab.txt')
    with open(vocab_path) as f:
        text = f.read()
        tokens = text.rstrip().split('\n')
    args_field.vocab = Vocab(tokens, specials=['<unk>', '<pad>', '<X>', '<Y>']) #, vectors='fasttext.en', vectors_cache='/fasttext')
    config.n_args = len(args_field.vocab)
    # relemb model
    relation_embedding_model = Pair2RelModel(config, args_field.vocab, args_field.vocab)
    load_model(model_file, relation_embedding_model)
    for param in relation_embedding_model.parameters():
        param.requires_grad = False
    relation_embedding_model.eval()
    relation_embedding_model.cuda()
    return relation_embedding_model

def get_batch(pairs, vocab):
    word1 = [vocab.stoi[w1] for w1, _ in pairs]
    word2 = [vocab.stoi[w2] for _, w2 in pairs]
    return Variable(torch.LongTensor(word1), requires_grad=False).cuda(), Variable(torch.LongTensor(word2), requires_grad=False).cuda()

def get_scores(relation_embedding, model):
    # return Variable(torch.randn(10, 8, 100004))
    bs, d = relation_embedding.size()
    prev_word_embed = model.relation_lm.sos_embedding.unsqueeze(0).unsqueeze(0).expand(bs, 1, model.relation_lm.sos_embedding.size(-1))
    scores = []
    hidden = None #torch.zeros(1, bs,100).cuda() 
    for step in range(0, 8):
        inp = torch.cat((prev_word_embed,relation_embedding.unsqueeze(1)), -1)
        # import ipdb
        # ipdb.set_trace()
        output, hidden = model.relation_lm.contextualizer(inp.transpose(1,0)) if hidden is None else  model.relation_lm.contextualizer(inp.transpose(1,0), hidden)
        step_scores = log_softmax(model.relation_lm.decoder(output.transpose(1,0)), -1)
        scores += [step_scores]
        best_scores, best_idxs = torch.max(step_scores, -1)
        prev_word_embed = model.relation_lm.embedding(best_idxs.squeeze(1)).unsqueeze(1)
    return torch.cat(scores, 1)

def get_relation_embedding(word1, word2, model):
    embedded_subjects = model.represent_left_argument(word1)
    embedded_objects = model.represent_right_argument(word2)
    predicted_relations = model.predict_relations(embedded_subjects, embedded_objects)
    return predicted_relations


def get_rel_phrases(scores, model):
    vocab = model.rel_vocab
    best_scores, best_idxs = torch.max(scores, -1)
    best_idxs = best_idxs.cpu().data.numpy()
    best_scores = best_scores.cpu().data.numpy()
    rels, scores = [], []
    for best_idx, best_score in zip(best_idxs, best_scores):
        phrase = [vocab.itos[i] for i in best_idx]
        phrase = ' '.join([phrase[i] for i in range(len(phrase))])
        rels += [phrase]
        scores += [best_score.tolist()]
    return rels, scores

def print_best_rels(config_file, model_file, pairs, bs=10):
    model = get_model(config_file, model_file)
    rel_phrases, wscores = [], []
    for i in range(0, len(pairs), bs):
        word1, word2 = get_batch(pairs[i:i+bs], model.arg_vocab)
        relation_embedding = get_relation_embedding(word1, word2, model)
        scores = get_scores(relation_embedding, model)
        batch_rel_phrases, batch_scores = get_rel_phrases(scores, model)
        rel_phrases += batch_rel_phrases
        wscores += batch_scores
    for pair, relation, score in zip(pairs, rel_phrases, wscores):
        print(pair, '-->', relation, '-->', score)
    return pairs



if __name__ == '__main__':
    config_file = sys.argv[1]
    model_file = sys.argv[2]
    pairs = [('cheap', 'expensive'), ('portland', 'oregon'), ('animals', 'dogs'), ('portland', 'california'), ('monet', 'painter')]
    print_best_rels(config_file, model_file, pairs)
