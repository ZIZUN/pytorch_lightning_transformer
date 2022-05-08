import torch.nn as nn
import torch
import math

from .embedding import PosEncoding
from .attention import MultiHeadAttention


class TransformerDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config                
                
        self.word_embedding = nn.Embedding(config.vocab_size, config.transformer_hidden_size)
        self.pos_embedding = PosEncoding(config)
        self.decoder_layers = nn.ModuleList([TransformerDecoderLayer(config) for i in range(config.encoder_layer_num)])

    def forward(self, input_ids, enc_output, enc_attention_mask):
        
        input_repre = self.word_embedding(input_ids)
        input_repre += self.pos_embedding(input_repre.size(1))
        

        for layer in self.decoder_layers:
            input_repre = layer(input=input_repre, enc_output=enc_output, enc_attention_mask=enc_attention_mask)
            
        output = input_repre    
        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # input, attention_mask=None, encoder_output=None
        self.masked_attention = MultiHeadAttention(config)
        self.enc_dec_cross_attention = MultiHeadAttention(config)
        self.layernorm = nn.LayerNorm(config.transformer_hidden_size)
        
        self.linear_1 = nn.Linear(config.transformer_hidden_size, config.transformer_hidden_size * 4)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(config.transformer_hidden_size * 4, config.transformer_hidden_size)
        
    def forward(self, input, enc_output, enc_attention_mask):
        
        masked_mha_output = self.layernorm(input + self.masked_attention(input=input, 
                                                                             attention_mask=None, 
                                                                             encoder_output=None))
        
        cross_mha_output = self.layernorm(masked_mha_output + self.enc_dec_cross_attention(input=masked_mha_output,
                                                                                        attention_mask=enc_attention_mask,
                                                                                        encoder_output=enc_output))
        layer_output = self.layernorm(cross_mha_output + self.linear_2(self.relu(self.linear_1(cross_mha_output))))
        
        return layer_output