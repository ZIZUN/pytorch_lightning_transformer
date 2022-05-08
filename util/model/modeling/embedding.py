import torch.nn as nn
import torch
import math


class PosEncoding(nn.Module):
    def __init__(self, config):
        super().__init__()
        max_len = config.position_encoding_maxlen
        pos_encoding  = torch.ones(max_len, config.transformer_hidden_size)
        pos_encoding.requires_grad = False
        
        pos = torch.arange(0, max_len).unsqueeze(1)
        
        base_term = 10000 * torch.ones(int(config.transformer_hidden_size / 2))
        divide_term = torch.pow(base_term, torch.arange(0,config.transformer_hidden_size, 2) / config.transformer_hidden_size)
        
        pos_encoding[:,0::2] = torch.sin(pos / divide_term)
        pos_encoding[:,1::2] = torch.cos(pos / divide_term)
        
        pos_encoding = pos_encoding.unsqueeze(0)
        
        self.register_buffer('pos_encoding', pos_encoding)
        
    def forward(self, seq_len):       
        
        return self.pos_encoding[:,:seq_len,:]