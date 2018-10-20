import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Module):

    def __init__(self, d_in, d_out, dropout):
        super(Linear, self).__init__()
        self.linear = nn.Linear(d_in, d_out)
        self.dropout = nn.Dropout(p=dropout, inplace=True) if dropout > 0 else None

    def forward(self, x):
        if self.dropout:
            sizes = [x.size(0)] + ([1] * (len(x.size()) - 2)) + [x.size(-1)]
            mask = self.dropout(torch.ones(sizes, device=x.device))
            x = mask * x
        return self.linear(x)


class MLP(nn.Module):

    def __init__(self, d, dropout):
        super(MLP, self).__init__()
        linear1 = Linear(d, d, dropout)
        linear2 = Linear(d, d, dropout)
        self.mlp = nn.Sequential(linear1, nn.ReLU(), linear2)

    def forward(self, x):
        return self.mlp(x)


class ResidualMLP(nn.Module):

    def __init__(self, d, dropout):
        super(ResidualMLP, self).__init__()
        self.mlp = MLP(d, dropout)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        return self.norm(x + self.mlp(x))

