# Minimalist GPT model with a fixed architecture

import torch
import torch.nn as nn
import torch.nn.functional as F


class GPT(nn.Module): 
    def __init__(self, d_embed, n_head, num_layers, vocab_size, block_size):
        super().__init__()
        self.d_embed = d_embed
        self.n_head = n_head
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.wte = nn.Embedding(self.vocab_size, self.d_embed)
        self.wpe = nn.Embedding(self.block_size, self.d_embed)
        self.dropout = nn.Dropout(0.1)
        self.causal_mask = nn.Transformer.generate_square_subsequent_mask(sz = self.block_size)
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
        
    def forward(self, idx, targets = None):
    	b, t = idx.size()
    	assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
    	pos = torch.arange(0, t, dtype=torch.long).unsqueeze(0) # shape (1, t)
    	te = self.wte(idx) 
    	pe = self.wpe(pos)
    	x = te + pe
    	x = self.transformer_encoder(x, mask = self.causal_mask, is_causal = True)
    	x = self.layer_norm(x)
    	logits = self.lmhead(x)
    	loss = None
    	if targets is not None:
        	loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
    	return logits, loss


    @torch.no_grad()
    def generate(self, idx, num_tokens, do_sample): 
        # generate num_tokens new tokens, using self.block_size last 
        for _ in range(num_tokens):
        	idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
        	token_logits, _ = self(idx_cond)
        	token_logits = token_logits[:, -1, :]
        	probs = F.softmax(token_logits, dim = -1)
        	if do_sample:
        		idx_next = torch.multinomial(probs, num_samples=1)
        	else:
        		_, idx_next = torch.topk(probs, k=1, dim=-1)
        	idx = torch.cat((idx, idx_next), dim=1)
        return idx