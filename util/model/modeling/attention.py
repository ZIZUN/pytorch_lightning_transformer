import torch.nn as nn
import torch
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.query_linear = nn.Linear(config.transformer_hidden_size, config.qkv_hidden_size * config.multi_head_num)
        self.key_linear = nn.Linear(config.transformer_hidden_size, config.qkv_hidden_size * config.multi_head_num)
        self.value_linear = nn.Linear(config.transformer_hidden_size, config.qkv_hidden_size * config.multi_head_num)
        
        self.softmax = nn.Softmax(dim=-1)
        self.mha_linear = nn.Linear(config.transformer_hidden_size, config.transformer_hidden_size)

    def forward(self, input, attention_mask=None, encoder_output=None):
        q = self.query_linear(input)
        if encoder_output == None:
            k = self.key_linear(input)
            v = self.value_linear(input)
        else:
            k = self.key_linear(encoder_output)
            v = self.value_linear(encoder_output) 
               
        bsz= q.size(0)
        seq_len = k.size(1)
        
        q = q.view(bsz, self.config.multi_head_num, seq_len, -1)
        k = k.view(bsz, self.config.multi_head_num, seq_len, -1)
        v = v.view(bsz, self.config.multi_head_num, seq_len, -1)
        
        qk_mul = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(q.size(-1))
        
        if attention_mask != None: # encoder att or decoder cross att
            mask = attention_mask.unsqueeze(1).expand(bsz, seq_len, seq_len).unsqueeze(1)
        elif encoder_output == None and attention_mask == None: # decoder masked att
            mask = torch.ones(bsz,seq_len,seq_len)
            mask = mask.triu(diagonal=1)
            mask = mask.unsqueeze(1).to(qk_mul.device)      
        masked_qk_mul = qk_mul.masked_fill(mask == 1, -float('inf'))
        
        qk_score = self.softmax(masked_qk_mul) # divide by scaling factor
        attn_output = torch.matmul(qk_score, v)
        mha_output = self.mha_linear(attn_output.view(bsz, seq_len, -1))
        
        return mha_output