# Minimalist GPT model with a fixed architecture
# 3 layes
# 3 heads
# d_embed = 12
# layerNormBefore: True 

import torch
import torch.nn as nn
import torch.nn.functional as F


class BERT(nn.Module): 
    def __init__(self, d_embed, n_head, num_layers, vocab_size, seq_length):
        super().__init__()
        self.d_embed = d_embed
        self.n_head = n_head
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.wte = nn.Embedding(self.vocab_size + 1, self.d_embed)
        self.wpe = nn.Embedding(self.seq_length, self.d_embed)
        self.dropout = nn.Dropout(0.1)
        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model = self.d_embed,
        														nhead=self.n_head,
        														dim_feedforward=self.d_embed * 4,
        														activation='gelu',
        														batch_first = True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer = transformer_encoder_layer,
        												num_layers = self.num_layers,
        												norm=nn.LayerNorm(self.d_embed))
        self.layer_norm = nn.LayerNorm(self.d_embed)
        self.lmhead = nn.Linear(self.d_embed, self.vocab_size)
        
    def forward(self, idx, targets = None, mask = None):
        pos = torch.arange(self.seq_length)
        te = self.wte(idx) 
        pe = self.wpe(pos)
        x = te + pe
        x = self.transformer_encoder(x)
        x = self.layer_norm(x)
        logits = self.lmhead(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction = 'none')
            if mask is not None:
                loss = loss[mask.view(-1)]
                loss = loss.mean()
        return logits, loss


      
    @torch.no_grad()
    def generate_output(self, idx): 
        ''' Simply calls the forward method, and converts the logits into the most likely tokens'''
        x = self.forward(idx)[0]
        x = x.argmax(dim = 2)
        return x