import numpy as np
import torch
from torch.nn import Module, Linear, Dropout, Sequential, LSTM, Embedding, GRU, ReLU, Parameter
from embeddings.util import masked_softmax
from torch.autograd import Variable
from torch.nn.init import xavier_normal, constant
from embeddings.util import pretrained_embeddings_or_xavier
from embeddings.vocab import Vocab, Vectors

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
            print('Copied pretrained vectors into relation span representation')
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


class LSTMContextualizer(Module):
    def __init__(self, config):
        super(LSTMContextualizer, self).__init__()
        self.config = config
        bidirectional = getattr(config, 'bidirectional', True)
        self.rnn = LSTM(input_size=config.d_lstm_input, hidden_size=config.d_lstm_hidden, num_layers=config.n_lstm_layers, dropout=config.dropout, bidirectional=bidirectional)

    def forward(self, inputs):
        inputs = inputs.permute(1, 0, 2)
        outputs, _ = self.rnn(inputs )  # outputs: [seq_len, batch, hidden * 2]
        return outputs.permute(1, 0, 2)

