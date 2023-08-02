import math
import torch

import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(Embedding, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x):
        out = self.embed(x)
        return out
    

class PositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len, embed_model_dim):
        super(PositionalEmbedding, self).__init__()
        self.embed_dim = embed_model_dim

        pe = torch.zeros(max_seq_len, embed_model_dim)
        for pos in range(max_seq_len):
            for i in range(0, self.embed_dim, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/self.embed_dim)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/self.embed_dim)))
        pe = pe.unsqueeze(0)  # add a new dimension of size 1 at the pos 0
        self.register_buffer('pe', pe)

    def forward(self, x):

        # Make embeddings relatively larger
        x = x * math.sqrt(self.embed_dim)

        # Add constant to embedding
        seq_len = x.size(1)  # get the size of dim=1
        x = x + torch.autograd.Variable(self.pe[:,:seq_len], requires_grad=False)
        
        return x