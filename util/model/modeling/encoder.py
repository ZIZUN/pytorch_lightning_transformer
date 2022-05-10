import torch.nn as nn
import torch
import math
from .embedding import PosEncoding
from .attention import MultiHeadAttention


class TransformerEncoder(nn.Module):
    def __init__(self, config, shared_word_embedding):
        super().__init__()
        self.config = config                
                
        self.word_embedding = shared_word_embedding
        self.pos_embedding = PosEncoding(config)
        self.encoder_layers = nn.ModuleList([TransformerEncoderLayer(config) for i in range(config.encoder_layer_num)])

    def forward(self, input_ids, attention_mask):
        
        input_repre = self.word_embedding(input_ids)
        input_repre += self.pos_embedding(input_repre.size(1))

        for layer in self.encoder_layers:
            input_repre = layer(input=input_repre, attention_mask=attention_mask)
            
        output = input_repre
        return output
    
    
class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.multi_head_attention = MultiHeadAttention(config)
        self.layernorm = nn.LayerNorm(config.transformer_hidden_size)
        
        self.linear_1 = nn.Linear(config.transformer_hidden_size, config.transformer_hidden_size * 4)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(config.transformer_hidden_size * 4, config.transformer_hidden_size)
        
    def forward(self, input, attention_mask):
        mha_output = self.layernorm(input + self.multi_head_attention(input=input, attention_mask=attention_mask))
        layer_output = self.layernorm(mha_output + self.linear_2(self.relu(self.linear_1(mha_output))))
        
        return layer_output    