import torch.nn as nn
import torch
import math

class BertModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config                
                
        self.word_embedding = nn.Embedding(config.vocab_size, config.bert_hidden_size)
        self.pos_embedding = PosEncoding(config)
        self.bert_layers = nn.ModuleList([BertLayer(config) for i in range(config.bert_layer_num)])

    def forward(self, input_ids, attention_mask):
        
        input_repre = self.word_embedding(input_ids)
        input_repre += self.pos_embedding(input_repre.size(1))

        for layer in self.bert_layers:
            input_repre = layer(input=input_repre, attention_mask=attention_mask)
            
        output = input_repre            
            
        return output

class PosEncoding(nn.Module):
    def __init__(self, config):
        super().__init__()
        max_len = config.position_encoding_maxlen
        pos_encoding  = torch.ones(max_len, config.bert_hidden_size)
        pos_encoding.requires_grad = False
        
        pos = torch.arange(0, max_len).unsqueeze(1)
        
        base_term = 10000 * torch.ones(int(config.bert_hidden_size / 2))
        divide_term = torch.pow(base_term, torch.arange(0,config.bert_hidden_size, 2) / config.bert_hidden_size)
        
        pos_encoding[:,0::2] = torch.sin(pos / divide_term)
        pos_encoding[:,1::2] = torch.cos(pos / divide_term)
        
        pos_encoding = pos_encoding.unsqueeze(0)
        
        self.register_buffer('pos_encoding', pos_encoding)
        
    def forward(self, seq_len):       
        
        return self.pos_encoding[:,:seq_len,:]


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.multi_head_attention = MultiHeadAttention(config)
        self.layernorm = nn.LayerNorm(config.bert_hidden_size)
        
        self.linear_1 = nn.Linear(config.bert_hidden_size, config.bert_hidden_size * 4)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(config.bert_hidden_size * 4, config.bert_hidden_size)
        
    def forward(self, input, attention_mask):
        mha_output = self.layernorm(input + self.multi_head_attention(input=input, attention_mask=attention_mask))
        layer_output = self.layernorm(mha_output + self.linear_2(self.relu(self.linear_1(mha_output))))
        
        return layer_output


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.query_linear = nn.Linear(config.bert_hidden_size, config.qkv_hidden_size * config.multi_head_num)
        self.key_linear = nn.Linear(config.bert_hidden_size, config.qkv_hidden_size * config.multi_head_num)
        self.value_linear = nn.Linear(config.bert_hidden_size, config.qkv_hidden_size * config.multi_head_num)
        
        self.softmax = nn.Softmax(dim=-1)
        self.mha_linear = nn.Linear(config.bert_hidden_size, config.bert_hidden_size)

    def forward(self, input, attention_mask):
        q = self.query_linear(input)
        k = self.key_linear(input)
        v = self.value_linear(input)
        
        bsz= q.size(0)
        seq_len = k.size(1)
        
        q = q.view(bsz, self.config.multi_head_num, seq_len, -1)
        k = k.view(bsz, self.config.multi_head_num, seq_len, -1)
        v = v.view(bsz, self.config.multi_head_num, seq_len, -1)
        
        qk_mul = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(q.size(-1))
        mask = attention_mask.unsqueeze(1).expand(bsz, seq_len, seq_len).unsqueeze(1)
        
        qk_mul = qk_mul.masked_fill(mask == 1, -float('inf'))

        qk_score = self.softmax(qk_mul) # divide by scaling factor
      
        
        attn_output = torch.matmul(qk_score, v)
        mha_output = self.mha_linear(attn_output.view(bsz, seq_len, -1))        
        
        return mha_output
        
class BertConfig:
    def __init__(self):
        self.vocab_size = 50265
        self.bert_hidden_size = 768
        self.multi_head_num = 12
        self.position_encoding_maxlen = 512
        
        self.qkv_hidden_size = 64
                
        self.bert_layer_num = 12



# Test
if __name__ == "__main__":
            
    model_config = BertConfig()
    model = BertModel(config=model_config)

    input_ids_rand = torch.randint(0, 10, (5, 30))
    attention_mask = torch.randint(0, 2, (5, 30))

    output = model(input_ids=input_ids_rand, attention_mask=attention_mask)
