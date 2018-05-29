import torch
from torch.nn import Module, Linear, Dropout, Sequential, LSTM, Embedding, GRU, ReLU
from torch.nn.functional import softmax
from allennlp.nn.util import masked_softmax
from torch.autograd import Variable
from torch.nn.init import xavier_normal, constant
from noallen.util import pretrained_embeddings_or_xavier

class SpanRepresentation(Module):
    def __init__(self, config, d_output, vocab):
        super(SpanRepresentation, self).__init__()
        self.config = config
        self.vocab = vocab
        n_input =  len(vocab)
        self.embedding = Embedding(n_input, config.d_embed)


        self.contextualizer = LSTMContextualizer(config) if config.n_lstm_layers > 0 else lambda x : x
        self.dropout = Dropout(p=config.dropout)
        self.head_attention = Sequential(self.dropout, Linear(2 * config.d_lstm_hidden, 1))
        self.head_transform = Sequential(self.dropout, Linear(2 * config.d_lstm_hidden, d_output))
        self.init()

    def init(self):
        [xavier_normal(p) for p in self.parameters() if len(p.size()) > 1]
        if self.vocab.vectors is not None:
            self.embedding.weight.data.copy_(self.vocab.vectors)
        else:
            #xavier_normal(self.embedding.weight.data)
            self.embedding.reset_parameters()

    def forward(self, inputs):
        text, mask = inputs
        text = self.dropout(self.embedding(text))
        text = self.contextualizer(text)
        weights = masked_softmax(self.head_attention(text).squeeze(-1), mask.float())
        print(weights)
        representation = (weights.unsqueeze(2) * self.head_transform(text)).sum(dim=1)
        return representation

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
