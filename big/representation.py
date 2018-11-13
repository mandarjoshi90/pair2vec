import numpy as np
import sentencepiece as spm
import torch
from torch.nn import Module, Linear, Dropout, Sequential, LSTM, Embedding, GRU, ReLU, Parameter, LayerNorm
from torch.nn.functional import softmax, normalize, embedding, tanh, relu
from allennlp.nn.util import masked_softmax
from torch.autograd import Variable
from torch.nn.init import xavier_normal, constant
from big.util import pretrained_embeddings_or_xavier
from big.torchtext.vocab import Vocab
from big.torchtext.vocab import Vectors
from big.contextualizer import Transformer
import big.mlp as mlp

class SpanRepresentation(Module):

    def __init__(self, config, d_output, vocab):
        super(SpanRepresentation, self).__init__()
        self.config = config
        self.vocab = vocab
        n_input =  len(vocab)
        self.embedding = Embedding(n_input, config.d_embed, sparse=True)
        self.normalize_pretrained = getattr(config, 'normalize_pretrained', False)
        self.contextualizer = Transformer(config.d_embed, config.num_transformer_heads, config.num_transformer_layers, config.dropout, config.num_positions)
        #self.mlp = mlp.MLP(config.d_embed, config.dropout)
        self.linear1 = mlp.Linear(config.d_embed*2, config.d_embed*8, config.dropout)
        self.linear2 = mlp.Linear(config.d_embed*8, config.d_embed, config.dropout)
        self.init()
    
    def init(self):
        [xavier_normal(p) for p in self.parameters() if len(p.size()) > 1]
        if self.vocab.vectors is not None:
            pretrained = normalize(self.vocab.vectors, dim=-1) if self.normalize_pretrained else self.vocab.vectors
            self.embedding.weight.data.copy_(pretrained)
            print('Copied pretrained vectors into relation span representation')
        else:
            self.embedding.reset_parameters()
    
    def forward(self, text):
        mask = text.sign().float()
        v = self.embedding(text)
        v = self.contextualizer(v, mask)
        x_indices = (text == 2).nonzero()
        y_indices = (text == 3).nonzero()
        x = v[x_indices[:, 0], x_indices[:, 1], :]
        y = v[y_indices[:, 0], y_indices[:, 1], :]
        #x = torch.max(text + torch.log(mask.unsqueeze(-1)), dim=1)[0]
        #return self.mlp(tanh(x))
        mlp1 = self.linear1(torch.cat([x, y], dim=-1))
        mlp2 = self.linear2(relu(mlp1))
        return mlp2


class RelationLM(Module):
    def __init__(self, config, vocab):
        super(RelationLM, self).__init__()
        self.vocab = vocab
        self.config = config
        self.embedding_size = 100
        self.embedding = Embedding(len(self.vocab), self.embedding_size, sparse=True)
        self.normalize_pretrained = getattr(config, 'normalize_pretrained', False)
        # self.contextualizer = LSTMContextualizer(config) if config.n_lstm_layers > 0 else lambda x : x
        # self.contextualizer = LSTM(input_size=config.d_lstm_input , hidden_size=config.d_lstm_hidden, num_layers=config.n_lstm_layers, dropout=config.dropout, bidirectional=False)
        self.contextualizer = LSTM(400 , self.embedding_size, num_layers=config.n_lstm_layers, dropout=config.dropout, bidirectional=False)
        # self.contextualizer = Linear(config.d_lstm_input, config.d_lstm_hidden)
        self.decoder = Linear(self.embedding_size, len(vocab))
        self.sos_embedding = Parameter(torch.randn(self.embedding_size, out=self.embedding.weight.data.new()))
        if config.tie_weights:
            self.decoder.weight = self.embedding.weight
        self.init()

    def init(self):
        [xavier_normal(p) for p in self.parameters() if len(p.size()) > 1]
        if self.vocab.vectors is not None:
            pretrained = normalize(self.vocab.vectors, dim=-1) if self.normalize_pretrained else self.vocab.vectors
            self.embedding.weight.data.copy_(pretrained[:,:self.embedding_size])
            print('Copied pretrained vectors into relation span representation')
        else:
            self.embedding.reset_parameters()

    def forward(self, predicted_rel_embed, observed_relations):
        text, mask = observed_relations
        rel_word_embeddings = self.embedding(text)
        bs, seq_len, _ = rel_word_embeddings.size()
        sos = self.sos_embedding.unsqueeze(0).unsqueeze(0).expand(bs, 1, self.sos_embedding.size(-1))
        rel_word_embeddings = torch.cat((sos, rel_word_embeddings[:, :seq_len - 1, :]), 1)
        rnn_input = torch.cat((rel_word_embeddings, predicted_rel_embed.unsqueeze(1).expand(bs, seq_len, predicted_rel_embed.size(-1))), -1)
        rnn_output = self.contextualizer(rnn_input.permute(1,0,2))[0].permute(1,0,2)
        # rnn_output = self.contextualizer(rnn_input)
        scores = self.decoder(rnn_output)
        return scores


def get_subword_vocab(vocab_file):
    itos = []
    with open(vocab_file, encoding='utf-8') as f:
        for line in f:
            itos.append(line.strip())
    return Vocab(itos, specials=['<unk>'])

def get_word_to_bpe_subwords(word_vocab, subword_vocab, sp):
    mapping = []
    max_len = 0
    for word_idx, word in enumerate(word_vocab.itos):
        mapping.append([])
        if word in word_vocab.specials:
            continue
        subwords = sp.EncodeAsPieces(word)[:5]
        subwords = [subw.decode('utf-8') for subw in subwords]
        subwords = [subw for subw in subwords if subword_vocab.stoi[subw] != 0]
        for subword in subwords:
            mapping[word_idx].append(subword_vocab.stoi[subword])
        max_len = max_len if len(mapping[word_idx]) < max_len else len(mapping[word_idx])
    print(len(word_vocab), max_len)
    numpy_map = np.zeros((len(word_vocab), max_len), dtype=int)
    for i, subword_list in enumerate(mapping):
        for j, subword_idx in enumerate(subword_list):
            numpy_map[i, j] = subword_idx
    return torch.from_numpy(numpy_map)

class SubwordEmbedding(Module):
    def __init__(self, config, vocab):
        super(SubwordEmbedding, self).__init__()
        self.word_embedding = Embedding(len(vocab), config.d_embed, sparse=True)
        self.subword_vocab = get_subword_vocab(config.subword_vocab_file)
        init_with_pretrained = getattr(config, 'init_with_pretrained', True)
        if init_with_pretrained:
            vectors, vectors_cache = (None, None) if not init_with_pretrained else (getattr(config, 'subword_vecs', 'en.wiki.bpe.op5000.d100.w2v.txt'), getattr(config, 'subword_vecs_cache', 'data/bpemb'))
            self.subword_vocab.load_vectors(Vectors(vectors, vectors_cache))
        self.subword_embedding = Embedding(len(self.subword_vocab), config.subd_embed, sparse=True)
        self.config = config
        self.word_vocab = vocab
        sp = spm.SentencePieceProcessor()
        sp.Load(config.bpe_model_file)
        lens = [len(subw) for subw in self.subword_vocab.itos[1:]]
        minn, maxn = min(lens), max(lens)
        self.word_to_subwords = get_word_to_bpe_subwords(vocab, self.subword_vocab, sp).cuda()
        merge_fn_str = getattr(config, 'subword_merge_function', 'add')
        if merge_fn_str == 'add':
            self.merge = lambda x,y : x + y
        elif merge_fn_str == 'cat':
            self.merge = lambda x,y : torch.cat((x,y), -1)
        else:
            raise NotImplementedError()
        self.init()

    def init(self):
        [xavier_normal(p) for p in self.parameters() if len(p.size()) > 1]
        if self.word_vocab.vectors is not None:
            pretrained = normalize(self.word_vocab.vectors, dim=-1) if self.config.normalize_pretrained else self.word_vocab.vectors
            self.word_embedding.weight.data.copy_(pretrained)
            print('Copied pretrained word vecs for argument matrix')
        else:
            self.word_embedding.reset_parameters()
        if self.subword_vocab.vectors is not None:
            pretrained = normalize(self.subword_vocab.vectors, dim=-1) if self.config.normalize_pretrained else self.subword_vocab.vectors
            self.subword_embedding.weight.data.copy_(pretrained)
            print('Copied pretrained subword vecs for argument matrix')
        else:
            self.subword_embedding.reset_parameters()

    def forward(self, word):
        word_embedding = self.word_embedding(word)
        subword_embedding_seq = Variable(torch.index_select(self.word_to_subwords, 0, word.data), requires_grad=False)
        subword_embeddings = self.subword_embedding(subword_embedding_seq)
        # import ipdb
        # ipdb.set_trace()
        mask = 1 - torch.eq(subword_embedding_seq, 0).float()
        subword_embeddings = subword_embeddings * mask.unsqueeze(2).expand_as(subword_embeddings)
        final_embedding = self.merge(word_embedding, subword_embeddings.sum(1))
        return final_embedding


class PositionalRepresentation(Module):
    def __init__(self, config, vocab):
        super(PositionalRepresentation, self).__init__()
        self.num_positions = config.num_positions
        self.vocab = vocab
        self.pos_embeddings = Embedding(self.num_positions, config.d_pos, sparse=True)
        self.embedding = Embedding(len(vocab), config.d_embed, sparse=True)
        self.nonlinearity = ReLU()
        self.mlp = Sequential(Linear(config.d_pos + config.d_embed, config.d_embed), self.nonlinearity, Linear(config.d_embed, config.d_rels))

    def forward(self, inputs):
        position = inputs / len(self.vocab)
        word = inputs % len(self.vocab)
        word_embed = self.embedding(word)
        pos_embedding = self.pos_embeddings(position)
        relation = self.mlp(torch.cat((word_embed, pos_embedding), -1))
        return relation

    def reset_parameters(self):
        [xavier_normal(p) for p in self.parameters() if len(p.size()) > 1]
        if self.vocab.vectors is not None:
            self.embedding.weight.data.copy_(self.vocab.vectors)
        else:
            #xavier_normal(self.embedding.weight.data)
            self.embedding.reset_parameters()

class LRPositionalRepresentation(Module):
    def __init__(self, config, vocab):
        super(LRPositionalRepresentation, self).__init__()
        self.num_positions = config.num_positions
        self.vocab = vocab
        self.pos_embeddings = Embedding(self.num_positions, config.d_pos, sparse=True)
        self.mid_embedding = Embedding(len(vocab), config.d_embed, sparse=True)
        self.left_embedding = Embedding(len(vocab), config.d_embed, sparse=True)
        self.right_embedding = Embedding(len(vocab), config.d_embed, sparse=True)
        self.nonlinearity = ReLU()
        self.mlp = Sequential(Linear(config.d_pos + config.d_embed*3, config.d_embed), self.nonlinearity, Linear(config.d_embed, config.d_rels))

    def forward(self, inputs):
        # inputs = inputs[0]
        # import ipdb
        # ipdb.set_trace()
        left_ctx = inputs[:, 0] 
        right_ctx = inputs[:, 1] 
        position = inputs[:, 2] / len(self.vocab)
        word = inputs[:, 2] % len(self.vocab)
        left_embed, right_embed = self.left_embedding(left_ctx), self.right_embedding(right_ctx)
        word_embed = self.mid_embedding(word)
        pos_embedding = self.pos_embeddings(position)
        relation = self.mlp(torch.cat((left_embed, right_embed, word_embed, pos_embedding), -1))
        return relation

    def reset_parameters(self):
        [xavier_normal(p) for p in self.parameters() if len(p.size()) > 1]
        if self.vocab.vectors is not None:
            self.mid_embedding.weight.data.copy_(self.vocab.vectors)
            self.left_embedding.weight.data.copy_(self.vocab.vectors)
            self.right_embedding.weight.data.copy_(self.vocab.vectors)
        else:
            #xavier_normal(self.embedding.weight.data)
            self.left_embedding.reset_parameters()
            self.right_embedding.reset_parameters()
            self.mid_embedding.reset_parameters()

