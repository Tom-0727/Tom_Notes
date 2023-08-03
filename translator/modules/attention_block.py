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
        self.Q = nn.Linear(self.embed_dim, self.n_heads * self.dk, bias=False)
        self.K = nn.Linear(self.embed_dim, self.n_heads * self.dk, bias=False)
        self.V = nn.Linear(self.embed_dim, self.n_heads * self.dk, bias=False)

        self.out = nn.Linear(self.n_heads * self.dk, self.embed_dim)

    def forward(self, key, query, value, mask = None, cuda = True):
        
        # Get dim info
        batch_size = key.size(0)
        seq_len = key.size(1)

        # query dimension could change in decoder during inference
        seq_len_query = query.size(1)

        # get Q, K, V with size of (batch_size x n_heads x seq_len x dk)
        K = self.K(key).view(batch_size, seq_len, self.n_heads, self.dk).transpose(1, 2)
        Q = self.Q(query).view(batch_size, seq_len_query, self.n_heads, self.dk).transpose(1, 2)
        V = self.V(value).view(batch_size, seq_len, self.n_heads, self.dk).transpose(1, 2)

        # computes attention
        K_T = K.transpose(-1, -2)  # (batch_size, n_heads, dk, k_seq_len)
        product = torch.matmul(Q, K_T)/math.sqrt(self.dk)  # (batch_size, n_heads, q_seq_len, k_seq_len)

        if mask is not None:
            if cuda:
                mask = mask.to('cuda')
                product = product.masked_fill(mask == 0, float(-1e20))

        scores = torch.matmul(F.softmax(product, dim=-1), V)  # (batch_size, n_heads, q_seq_len, dk)

        # concatenate heads and put through final linear layer
        # (batch_size x num_heads x seq_len x dk) -> (batch_size x seq_len x num_heads x dk)  -> (batch_size, seq_len, d_model)
        concat = scores.transpose(1, 2).contiguous().view(batch_size, seq_len_query, self.dk*self.n_heads)

        output = self.out(concat)

        return output
