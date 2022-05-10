# pytorch_lightning_transformer
Transformer implementation from scratch

### Training Machine translation
```bash
python run.py # TODO
```

### Test
```python   
import torch
from util.model.modeling.transformer import Transformer, TransformerConfig
           
model_config = TransformerConfig()
model = Transformer(config=model_config)

enc_input_ids_rand = torch.randint(0, 10, (5, 30))
enc_attention_mask = torch.randint(0, 2, (5, 30))

dec_input_ids_rand = torch.randint(0, 10, (5, 30))


output = model(enc_input_ids=enc_input_ids_rand, 
               enc_attention_mask=enc_attention_mask,
               dec_input_ids=dec_input_ids_rand)
```

### Model Implementation
```python
class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config                

        self.shared_word_embedding = nn.Embedding(config.vocab_size, config.transformer_hidden_size)                
        self.encoder = TransformerEncoder(config, shared_word_embedding=self.shared_word_embedding)
        self.decoder = TransformerDecoder(config, shared_word_embedding=self.shared_word_embedding)

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
        
class TransformerDecoder(nn.Module):
    def __init__(self, config, shared_word_embedding):
        super().__init__()
        self.config = config                
                
        self.word_embedding = shared_word_embedding
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
        
        q = q.view(bsz, seq_len, self.config.multi_head_num, -1).transpose(1,2)
        k = k.view(bsz, seq_len, self.config.multi_head_num, -1).transpose(1,2)
        v = v.view(bsz, seq_len, self.config.multi_head_num, -1).transpose(1,2)
        
        qk_mul = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(q.size(-1))
        
        if attention_mask != None: # encoder att or decoder cross att
            mask = attention_mask.unsqueeze(1).expand(bsz, seq_len, seq_len).unsqueeze(1)
        elif encoder_output == None and attention_mask == None: # decoder masked att
            mask = torch.ones(bsz,seq_len,seq_len)
            mask = mask.triu(diagonal=1)
            mask = (mask==0).unsqueeze(1).to(qk_mul.device)   
        masked_qk_mul = qk_mul.masked_fill(mask == 0, -float('inf'))
        
        qk_score = self.softmax(masked_qk_mul) # divide by scaling factor
        attn_output = torch.matmul(qk_score, v)
        mha_output = self.mha_linear(attn_output.view(bsz, seq_len, -1))
        
        return mha_output
```
