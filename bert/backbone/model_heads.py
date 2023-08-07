import torch
import torch.nn as nn


# For Masked Language Model Task
class MLMHead(nn.Module):
    def __init__(self, embed_dim: int, vocab_size: int):
        super(MLMHead, self).__init__()

        # Config
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

        # Layers
        self.linear = nn.Linear(self.embed_dim, self.vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        opt = self.linear(x[:, 1:, :]) # batch_size x token_seq_len x embed_dim
        opt = self.softmax(opt)

        return opt
    

# For Next Sentence Prediction Task
class NSPHead(nn.Module):
    def __init__(self, embed_dim: int):
        super(NSPHead, self).__init__()

        # Config
        self.embed_dim = embed_dim

        # Layers
        self.linear = nn.Linear(self.embed_dim, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        opt = self.linear(x[:, 0, :]) # batch_size x [CLS] x embed_dim
        opt = self.softmax(opt)

        return opt