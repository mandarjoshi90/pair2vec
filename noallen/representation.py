import numpy as np
import sentencepiece as spm
import torch
from torch.nn import Module, Linear, Dropout, Sequential, LSTM, Embedding, GRU, ReLU
from torch.nn.functional import softmax, normalize
from allennlp.nn.util import masked_softmax
from torch.autograd import Variable
from torch.nn.init import xavier_normal, constant
from noallen.util import pretrained_embeddings_or_xavier
from noallen.torchtext.vocab import Vocab
from noallen.torchtext.vocab import Vectors

class SpanRepresentation(Module):
    def __init__(self, config, d_output, vocab):
        super(SpanRepresentation, self).__init__()
        self.config = config
        self.vocab = vocab
        n_input =  len(vocab)
        self.embedding = Embedding(n_input, config.d_embed)
        self.normalize_pretrained = getattr(config, 'normalize_pretrained', False)


        self.contextualizer = LSTMContextualizer(config) if config.n_lstm_layers > 0 else lambda x : x
        self.dropout = Dropout(p=config.dropout)
        self.head_attention = Sequential(self.dropout, Linear(2 * config.d_lstm_hidden, 1))
        self.head_transform = Sequential(self.dropout, Linear(2 * config.d_lstm_hidden, d_output))
        self.init()

    def init(self):
        [xavier_normal(p) for p in self.parameters() if len(p.size()) > 1]
        if self.vocab.vectors is not None:
            pretrained = normalize(self.vocab.vectors, dim=-1) if self.normalize_pretrained else self.vocab.vectors
            self.embedding.weight.data.copy_(pretrained)
        else:
            #xavier_normal(self.embedding.weight.data)
            self.embedding.reset_parameters()

    def forward(self, inputs):
        text, mask = inputs
        text = self.dropout(self.embedding(text))
        text = self.contextualizer(text)
        weights = masked_softmax(self.head_attention(text).squeeze(-1), mask.float())
        representation = (weights.unsqueeze(2) * self.head_transform(text)).sum(dim=1)
        return representation

def get_subword_vocab(vocab_file):
    itos = []
    with open(vocab_file, encoding='utf-8') as f:
        for line in f:
            itos.append(line.strip())
    return Vocab(itos, specials=['<unk>'])

# def get_word_to_subwords(word_to_subwords_file, word_vocab, subword_vocab):
    # mapping = [[]] * len(word_vocab)
    # max_len = 0
    # with open(word_to_subwords, encoding='utf-8') as f:
        # for line in f:
            # parts = line.strip().split('\t')
            # for subword in parts[1:]:
                # word_idx = word_vocab.stoi[parts[0]]
                # mapping[word_idx].append(subword_vocab.stoi[subword])
                # max_len = max_len if len(mapping[word_idx]) < max_len else len(mapping[word_idx])
    # numpy_map = np.zeros(len(word_vocab), max_len)
    # for i, subword_list in enumerate(mapping):
        # for j, subword_idx in enumerate(subword_list):
            # numpy_map[i, j] = subword_idx
    # return torch.from_(numpy_map)

def get_subwords(word, minn, maxn):
    ngrams = []
    word = '<' + word  + '>'
    for start in range(len(word)):
        for end in range(start+minn, min(start+maxn+1, len(word)+1)):
            if end - start < len(word):
                ngrams.append(word[start:end])
    return ngrams[:10]

def get_word_to_subwords(word_vocab, subword_vocab, minn, maxn):
    mapping = []
    max_len = 0
    for word_idx, word in enumerate(word_vocab.itos):
        mapping.append([])
        if word in word_vocab.specials:
            continue
        subwords = get_subwords(word, minn, maxn)
        subwords = [subw for subw in subwords if subword_vocab.stoi[subw] != 0]
        for subword in subwords:
            mapping[word_idx].append(subword_vocab.stoi[subword])
        max_len = max_len if len(mapping[word_idx]) < max_len else len(mapping[word_idx])
    print(len(word_vocab), max_len, minn, maxn)
    numpy_map = np.zeros((len(word_vocab), max_len), dtype=int)
    for i, subword_list in enumerate(mapping):
        for j, subword_idx in enumerate(subword_list):
            numpy_map[i, j] = subword_idx
    return torch.from_numpy(numpy_map)

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
        self.word_embedding = Embedding(len(vocab), config.d_embed)
        self.subword_vocab = get_subword_vocab(config.subword_vocab_file)
        self.subword_vocab.load_vectors(Vectors('en.wiki.bpe.op5000.d300.w2v.txt', '/home/mandar90/data/bpemb'))
        self.subword_embedding = Embedding(len(self.subword_vocab), config.d_embed) 
        self.config = config
        self.word_vocab = vocab
        sp = spm.SentencePieceProcessor()
        sp.Load(config.bpe_model_file)
        lens = [len(subw) for subw in self.subword_vocab.itos[1:]]
        minn, maxn = min(lens), max(lens)
        self.word_to_subwords = get_word_to_bpe_subwords(vocab, self.subword_vocab, sp).cuda()
        self.init()

    def init(self):
        [xavier_normal(p) for p in self.parameters() if len(p.size()) > 1]
        if self.word_vocab.vectors is not None:
            pretrained = normalize(self.word_vocab.vectors, dim=-1) if self.config.normalize_pretrained else self.word_vocab.vectors
            self.word_embedding.weight.data.copy_(pretrained)
        else:
            self.word_embedding.reset_parameters()
        if self.subword_vocab.vectors is not None:
            pretrained = normalize(self.subword_vocab.vectors, dim=-1) if self.config.normalize_pretrained else self.subword_vocab.vectors
            self.subword_embedding.weight.data.copy_(pretrained)
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
        return word_embedding + subword_embeddings.sum(1)


class PositionalRepresentation(Module):
    def __init__(self, config, vocab):
        super(PositionalRepresentation, self).__init__()
        self.num_positions = config.num_positions
        self.vocab = vocab
        self.pos_embeddings = Embedding(self.num_positions, config.d_pos)
        self.embedding = Embedding(len(vocab), config.d_embed)
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
        self.pos_embeddings = Embedding(self.num_positions, config.d_pos)
        self.mid_embedding = Embedding(len(vocab), config.d_embed)
        self.left_embedding = Embedding(len(vocab), config.d_embed)
        self.right_embedding = Embedding(len(vocab), config.d_embed)
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

class LSTMContextualizer(Module):
    def __init__(self, config):
        super(LSTMContextualizer, self).__init__()
        self.config = config
        self.rnn = LSTM(input_size=config.d_lstm_input, hidden_size=config.d_lstm_hidden, num_layers=config.n_lstm_layers, dropout=config.dropout, bidirectional=True)

    def forward(self, inputs):
        inputs = inputs.permute(1, 0, 2)
        #batch_size = inputs.size()[1]
        #state_shape = self.config.n_lstm_layers * 2, batch_size, self.config.d_lstm_hidden
        #h0 = c0 = Variable(inputs.data.new(*state_shape).zero_())
        #outputs, (ht, ct) = self.rnn(inputs )  # outputs: [seq_len, batch, hidden * 2]
        outputs, _ = self.rnn(inputs )  # outputs: [seq_len, batch, hidden * 2]
        return outputs.permute(1, 0, 2)
