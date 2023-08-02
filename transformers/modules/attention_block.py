import math
import torch

import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, 
                 embed_dim: int = 512, 
                 n_heads: int = 8):
        super(MultiHeadAttention, self).__init__()

        # Basic Attributes
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.dk = embed_dim // n_heads

        # Query, Key, Value : input_dim = d_model//n_heads = dk
        self.Q = nn.Linear(self.dk, self.dk, bias=False)
        self.K = nn.Linear(self.dk, self.dk, bias=False)
        self.V = nn.Linear(self.dk, self.dk, bias=False)
        self.out = nn.Linear(self.n_heads * self.dk, self.embed_dim)

    def forward(self, key, query, value, mask = None):
        
        # Get dim info
        batch_size = key.size(0)
        seq_length = key.size(1)

        # query dimension could change in decoder during inference
        seq_length_query = query.size(1)

        # (batch_size x seq_length x 8 x 64)
        key = key.view(batch_size, seq_length, self.n_heads, self.dk)
        query = query.view(batch_size, seq_length_query, self.n_heads, self.dk)
        value = value.view(batch_size, seq_length, self.n_heads, self.dk)

        k = self.K(key)
        q = self.Q(query)
        v = self.V(value)

        # (batch_size, n_heads, seq_len, dk)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # computes attention
        k_T = k.transpose(-1, -2)  # (batch_size, n_heads, dk, seq_len)
        product = torch.matmul(q, k_T)/math.sqrt(self.dk)

        if mask is not None:
            product = product.masked_fill(mask == 0, float(-1e20))

        scores = torch.matmul(F.softmax(product, dim=-1), v)

        # concatenate heads and put through final linear layer
        # (32x8x10x64) -> (32x10x8x64)  -> (batch_size, seq_len, d_model)
        concat = scores.transpose(1, 2).contiguous().view(batch_size, seq_length_query, self.dk*self.n_heads)

        output = self.out(concat)

        return output
