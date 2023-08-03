
import torch

import torch.nn as nn
import torch.nn.functional as F

from modules import *

class EncodeBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor=4, n_heads=8):
        super(EncodeBlock, self).__init__()

        self.attention = MultiHeadAttention(embed_dim, n_heads)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * expansion_factor),  # 512 * 2048
            nn.ReLU(),
            nn.Linear(embed_dim * expansion_factor, embed_dim),
        )

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, key, query, value):
        # Multi-head Attention
        attention_out = self.attention(key = key, 
                                       query = query, 
                                       value = value)
        
        # Notice that you need to add the query but not K/V because when you make Cross Attention the opt would be in size of query 
        attention_res_out = attention_out + query  # Residual connection
        norm1_out = self.dropout1(self.norm1(attention_res_out))  # Normalization & DropOut

        ff_out = self.feed_forward(norm1_out)
        ff_res_out = ff_out + norm1_out  # Residual connection
        norm2_out = self.dropout2(self.norm2(ff_res_out))  # Normalization & DropOut

        return norm2_out


class Encoder(nn.Module):
    def __init__(self, max_seq_len, vocab_size, embed_dim, num_layers=6, expansion_factor=4, n_heads=8):
        super(Encoder, self).__init__()

        self.embedding_layer = Embedding(vocab_size, embed_dim)  
        self.positional_encoder = PositionalEmbedding(max_seq_len, embed_dim)

        self.layers = nn.ModuleList([EncodeBlock(embed_dim, expansion_factor, n_heads) for _ in range(num_layers)])
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):

        # Word Embedding + Positional Encoding
        embed_out = self.embedding_layer(x)
        out = self.positional_encoder(embed_out)
        out = self.dropout(out)

        for layer in self.layers:
            out = layer(key = out, 
                        query = out, 
                        value = out)
        
        return out


class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor=4, n_heads=8):
        super(DecoderBlock, self).__init__()

        self.attention = MultiHeadAttention(embed_dim, n_heads)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
        self.transformer_block = EncodeBlock(embed_dim, expansion_factor, n_heads)

    def forward(self, key, value, x, mask):

        # Only pass mask to the first attention block
        attention = self.attention(key = x, 
                                   query = x, 
                                   value = x, 
                                   mask = mask)
        query = self.dropout(self.norm(attention + x))

        # Cross Attention
        out = self.transformer_block(key = key, 
                                     query = query, 
                                     value = value)

        return out
    

class Decoder(nn.Module):
    def __init__(self, t_vocab_size, embed_dim, max_seq_len, num_layers=6, expansion_factor=4, n_heads=8):
        super(Decoder, self).__init__()

        self.word_embedding = Embedding(t_vocab_size, embed_dim)
        self.pos_embedding = PositionalEmbedding(max_seq_len, embed_dim)
        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_dim, expansion_factor, n_heads)
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_dim, t_vocab_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, encoder_out, mask):
        x = self.word_embedding(x)
        x = self.pos_embedding(x)

        # dropout in each pos+embeddings & before each sub_layer
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(key = encoder_out, 
                      x = x, 
                      value = encoder_out, 
                      mask = mask)  # dropout had made in sub_layer

        out = F.softmax(self.fc_out(x), dim=-1)
        return out
    

class Transformer(nn.Module):
    def __init__(self, embed_dim, s_vocab_size, t_vocab_size, max_seq_len, 
                 num_layers = 6, 
                 expansion_factor = 4,
                 n_heads = 8):
        super(Transformer, self).__init__()

        self.t_vocab_size = t_vocab_size
        self.encoder = Encoder(max_seq_len = max_seq_len, 
                               vocab_size = s_vocab_size, 
                               embed_dim = embed_dim, 
                               num_layers = num_layers, 
                               expansion_factor = expansion_factor, 
                               n_heads = n_heads)
        self.decoder = Decoder(t_vocab_size = t_vocab_size, 
                               embed_dim = embed_dim, 
                               max_seq_len = max_seq_len, 
                               num_layers = num_layers, 
                               expansion_factor = expansion_factor, 
                               n_heads = n_heads)
    
    # Get the triangle mask for the target sequence
    def make_trg_mask(self, trg):
        batch_size, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(batch_size, 1, trg_len, trg_len)
        return trg_mask
    
    # For training
    def forward(self, src, trg):
        trg_mask = self.make_trg_mask(trg)
        encoder_out = self.encoder(src)
        outputs = self.decoder(trg, encoder_out, trg_mask)
        return outputs