# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F


class MultiheadAttention(nn.Module):

    def __init__(self, d, num_heads, dropout):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = d // num_heads
        assert self.head_dim * num_heads == d, "d must be divisible by num_heads"
        self.scaling = d ** -0.5
        
        self.in_proj = Linear(d, d * 3, dropout)
        self.out_proj = Linear(d, d, dropout)
        self.dropout = nn.Dropout(dropout)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.in_proj.bias, 0.)
        nn.init.constant_(self.out_proj.bias, 0.)

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
        
        y = heads.contiguous().view(b, self.num_heads, l, self.head_dim).transpose(1, 2).view(b, l, d)
        y = self.out_proj(y)
        return y

