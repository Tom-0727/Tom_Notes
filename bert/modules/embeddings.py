import math
import torch
import torch.nn as nn


class WordEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int):
        super(WordEmbedding, self).__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        self.embed = nn.Embedding(self.vocab_size, self.embed_dim)

    def forward(self, x):
        return self.embed(x)
    

class PositionEmbedding(nn.Module):
    def __init__(self, max_seq_len: int, embed_dim: int):
        super(PositionEmbedding, self).__init__()

        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim

        pe = torch.zeros(self.max_seq_len, self.embed_dim)
        for pos in range(self.max_seq_len):
            for i in range(self.embed_dim, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/self.embed_dim)))
                pe[pos, i+1] = math.cos(pos / (10000 ** (2 * i/self.embed_dim)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x
        
class SegmentEmbedding(nn.Module):
    def __init__(self, embed_dim: int):
        super(SegmentEmbedding, self).__init__()

        self.embed_dim = embed_dim

        self.embed = nn.Embedding(3, self.embed_dim)  # for SentenceA, SEP, SentenceB

    def forward(self, x):
        return self.embed(x)
    

class JointEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, max_seq_len: int):
        super(JointEmbeddings, self).__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # Joint Embeddings
        self.word_embed = WordEmbedding(self.vocab_size, self.embed_dim)
        self.pos_embed = PositionEmbedding(self.max_seq_len, self.embed_dim)
        self.seg_embed = SegmentEmbedding(self.embed_dim)

        # Dropout
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, segment_ids):
        word_embed = self.word_embed(x) * math.sqrt(self.embed_dim)
        pos_embed = self.pos_embed(x)
        seg_embed = self.seg_embed(segment_ids)

        joint_embed = (word_embed + pos_embed + seg_embed)

        return self.dropout(joint_embed)
