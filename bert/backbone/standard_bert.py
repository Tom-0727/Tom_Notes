import torch
import torch.nn as nn

from bert.modules import *
from .model_heads import *


class EncodeBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, expansion_factor: int):
        super(EncodeBlock, self).__init__()

        # Config
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.expansion_factor = expansion_factor

        # Layers
        self.attention = MultiHeadAttention(self.embed_dim, self.num_heads)
        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.dropout1 = nn.Dropout(0.1)

        self.feed_forward = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * self.expansion_factor),
            nn.GELU(),
            nn.Linear(self.embed_dim * self.expansion_factor, self.embed_dim)
        )
        self.norm2 = nn.LayerNorm(self.embed_dim)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, x):
        # Attention
        attention = self.attention(x)
        attention = self.norm1(attention + x)
        attention = self.dropout1(attention)

        # Feed Forward
        feed_forward = self.feed_forward(attention)
        feed_forward = self.norm2(feed_forward + attention)
        opt = self.dropout2(feed_forward)

        return opt


class BERTBackbone(nn.Module):
    def __init__(self, 
                 embed_dim: int, 
                 num_heads: int, 
                 expansion_factor: int, 
                 num_layers: int):
        super(BERTBackbone, self).__init__()

        # Config
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.expansion_factor = expansion_factor
        self.num_layers = num_layers

        # Layers
        self.layers = nn.ModuleList([EncodeBlock(self.embed_dim, self.num_heads, self.expansion_factor) for _ in range(self.num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class BERT(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 expansion_factor: int,
                 num_layers: int,
                 vocab_size: int):
        super(BERT, self).__init__()

        # Config
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.expansion_factor = expansion_factor
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        # Layers
        self.joint_embeddings = JointEmbeddings(self.embed_dim)
        self.backbone = BERTBackbone(self.embed_dim, self.num_heads, self.expansion_factor, self.num_layers)
        self.mlm_head = MLMHead(self.embed_dim, self.vocab_size)
        self.nsp_head = NSPHead(self.embed_dim)
    
    def forward(self, x):
        # Joint Embeddings
        joint_embeddings = self.joint_embeddings(x)

        # Backbone
        hidden_state = self.backbone(joint_embeddings)

        # MLM Head
        mlm_opt = self.mlm_head(hidden_state)

        # NSP Head
        nsp_opt = self.nsp_head(hidden_state)

        return mlm_opt, nsp_opt
