import torch
import torch.nn as nn
import torch.nn.functional as F

from big.mlp import Linear, MLP


class Transformer(nn.Module):
    
    def __init__(self, d, num_heads, num_layers, dropout, max_len):
        super(Transformer, self).__init__()
        self.positions = nn.Embedding(max_len, d)
        self.layers = nn.ModuleList([TransformerLayer(d, num_heads, dropout) for _ in range(num_layers)])

    def forward(self, x, mask):
        positions = self.positions.weight[:x.size(1)]
        x = x + positions
        mask = torch.log(mask)
        for layer in self.layers:
            x = layer(x, mask)
        return x


class TransformerLayer(nn.Module):

    def __init__(self, d, num_heads, dropout):
        super(TransformerLayer, self).__init__()
        self.att = MultiheadAttention(d, num_heads, dropout)
        self.mlp = MLP(d, dropout)
        self.att_norm = nn.LayerNorm(d)
        self.mlp_norm = nn.LayerNorm(d)

    def forward(self, x, mask):
        x = self.att_norm(x + self.att(x, mask))
        x = self.mlp_norm(x + self.mlp(x))
        return x


class MultiheadAttention(nn.Module):

    def __init__(self, d, num_heads, dropout):
        super(MultiheadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = d // num_heads
        self.scaling = d ** -0.5
        self.in_proj = Linear(d, d * 3, dropout)
        self.out_proj = Linear(d, d, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        b, l, d = x.size()
        x = self.in_proj(x.transpose(0, 1))
        x = x.contiguous().view(l, b * self.num_heads, self.head_dim * 3).transpose(0, 1)
        q, k, v = x.chunk(3, dim=-1)   # [b*h, l, hd]

        attention = torch.bmm(q * self.scaling, k.transpose(1, 2))   # [b*h, l, l]
        attention = attention.view(b, self.num_heads, l, l) + mask.view(b, 1, 1, l)
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        heads = torch.bmm(attention.view(b * self.num_heads, l, l), v)   # [b*h, l, hd]

        y = heads.contiguous().view(b, self.num_heads, l, self.head_dim).transpose(1, 2).contiguous().view(b, l, d)
        y = self.out_proj(y)
        return y

