import torch.nn as nn
import torch
import math
from .encoder import TransformerEncoder
from .decoder import TransformerDecoder


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config                
                
        self.encoder = TransformerEncoder(config)
        self.decoder = TransformerDecoder(config)

    def forward(self, enc_input_ids, enc_attention_mask, dec_input_ids):
        
        enc_output = self.encoder(input_ids=enc_input_ids, attention_mask=enc_attention_mask)
        dec_output = self.decoder(input_ids=dec_input_ids, enc_output=enc_output, enc_attention_mask=enc_attention_mask)
        
        return dec_output


class TransformerConfig:
    def __init__(self):
        self.vocab_size = 50265
        self.transformer_hidden_size = 512
        self.multi_head_num = 8
        self.position_encoding_maxlen = 512
        
        self.qkv_hidden_size = 64
                
        self.encoder_layer_num = 6
        self.decoder_layer_num = 6