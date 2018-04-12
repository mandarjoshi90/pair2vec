import torch
from torch.nn import Module, Linear, Dropout, Sequential, LSTM, Embedding
from torch.nn.functional import softmax
from allennlp.nn.util import masked_softmax
from torch.autograd import Variable
from torch.nn.init import xavier_normal_, constant_
from noallen.util import pretrained_embeddings_or_xavier

class SpanRepresentation(Module):
    
    def __init__(self, config, d_output, vocab, namespace):
        super(SpanRepresentation, self).__init__()
        self.config = config
        self.vocab = vocab
        self.vocab_namespace = namespace
        n_input = vocab.get_vocab_size(namespace)
        self.embedding = Embedding(n_input, config.d_embed)


        self.contextualizer = LSTMContextualizer(config) if config.n_lstm_layers > 0 else lambda x : x
        self.dropout = Dropout(p=config.dropout)
        self.head_attention = Sequential(self.dropout, Linear(2 * config.d_lstm_hidden, 1))
        self.head_transform = Sequential(self.dropout, Linear(2 * config.d_lstm_hidden, d_output))
        self.init()

    def init(self):
        [xavier_normal_(p) for p in self.parameters() if len(p.size()) > 1]
        pretrained_embeddings_or_xavier(self.config, self.embedding, self.vocab, self.vocab_namespace)


    def forward(self, inputs):
        text, mask = inputs
        text = self.contextualizer(self.embedding(text))
        weights = masked_softmax(self.head_attention(text).squeeze(-1), mask.float())
        representation = (weights.unsqueeze(2) * self.head_transform(text)).sum(dim=1)
        return representation


class LSTMContextualizer(Module):
    
    def __init__(self, config):
        super(LSTMContextualizer, self).__init__()
        self.config = config
        self.rnn = LSTM(input_size=config.d_lstm_input, hidden_size=config.d_lstm_hidden, num_layers=config.n_lstm_layers, dropout=config.dropout, bidirectional=True)
    
    def forward(self, inputs):
        inputs = inputs.permute(1, 0, 2)
        batch_size = inputs.size()[1]
        state_shape = self.config.n_lstm_layers * 2, batch_size, self.config.d_lstm_hidden
        h0 = c0 = Variable(inputs.data.new(*state_shape).zero_())
        outputs, (ht, ct) = self.rnn(inputs, (h0, c0))  # outputs: [seq_len, batch, hidden * 2]
        return outputs.permute(1, 0, 2)
