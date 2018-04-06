import torch
from torch.nn import Module, Linear, Dropout, Sequential, LSTM
from torch.nn.functional import softmax
from torch.autograd import Variable

class SpanRepresentation(Module):
    '''
    Assumes that input is (ELMo vectors, masks).
    '''
    
    def __init__(self, config, d_output):
        super(SpanRepresentation, self).__init__()
        self.contextualizer = LSTMContextualizer(config)
        self.dropout = Dropout(p=config.dropout)
        self.head_attention = Sequential(self.dropout, Linear(2 * config.d_lstm_hidden, 1))
        self.head_transform = Sequential(self.dropout, Linear(2 * config.d_lstm_hidden, d_output))
    
    def forward(self, inputs):
        text, mask = inputs
        text = self.contextualizer(text)
        mask = torch.log(mask).unsqueeze(-1)
        weights = softmax(self.head_attention(text) + mask)
        representation = (weights * self.head_transform(text)).sum(dim=1)
        return representation


class LSTMContextualizer(Module):
    
    def __init__(self, config):
        super(LSTMContextualizer, self).__init__()
        self.config = config
        self.input_size = config.d_proj if config.projection else config.d_embed
        self.output_size = config.d_hidden * 2 if config.birnn else config.d_hidden
        self.rnn = LSTM(input_size=self.d_lstm_input, hidden_size=config.d_lstm_hidden, num_layers=config.n_lstm_layers, dropout=config.dropout, bidirectional=True)
    
    def forward(self, inputs):
        inputs = inputs.permute(1, 0, 2)
        batch_size = inputs.size()[1]
        state_shape = self.config.n_cells, batch_size, self.config.d_hidden
        h0 = c0 = Variable(inputs.data.new(*state_shape).zero_())
        outputs, (ht, ct) = self.rnn(inputs, (h0, c0))  # outputs: [seq_len, batch, hidden * 2]
        return outputs.permute(1, 0, 2)
