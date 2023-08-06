import torch
import torch.nn as nn



class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, expansion_factor: int):
        super(MultiHeadAttention, self).__init__()

        # Structure
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.dk = embed_dim // n_heads

        # Q, K, V Layers
        self.Q = nn.Linear(self.embed_dim, self.embed_dim)
        self.K = nn.Linear(self.embed_dim, self.embed_dim)
        self.V = nn.Linear(self.embed_dim, self.embed_dim)

        # Output Layer
        self.out = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, x):
        # Get Info
        batch_size = x.size(0)
        seq_len = x.size(1)

        query = self.Q(x)  # (batch_size, seq_len, dk * n_heads)
        key = self.K(x)  # (batch_size, seq_len, dk * n_heads)
        value = self.V(x)  # (batch_size, seq_len, dk * n_heads)

        # Split into multiple heads -> # (batch_size, n_heads, seq_len, dk)
        query = query.view(batch_size, seq_len, self.n_heads, self.dk).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.n_heads, self.dk).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.n_heads, self.dk).transpose(1, 2)

        # Attention
        scores = torch.matmul(query, key.transpose(-1, -2)) / (self.dk ** 0.5)  # (batch_size, n_heads, seq_len, seq_len)
        attention = torch.matmul(torch.softmax(scores, dim=-1), value)  # (batch_size, n_heads, seq_len, dk)
        multi_head_attention = attention.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        # output
        opt = self.out(multi_head_attention)
        
        return opt