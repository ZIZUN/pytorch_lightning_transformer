import torch.nn as nn
import torch
import math
from util.model.modeling.transformer import Transformer, TransformerConfig

# Test
if __name__ == "__main__":
            
    model_config = TransformerConfig()
    model = Transformer(config=model_config)

    enc_input_ids_rand = torch.randint(0, 10, (5, 30))
    enc_attention_mask = torch.randint(0, 2, (5, 30))
    
    dec_input_ids_rand = torch.randint(0, 10, (5, 30))
    

    output = model(enc_input_ids=enc_input_ids_rand, 
                   enc_attention_mask=enc_attention_mask,
                   dec_input_ids=dec_input_ids_rand)
    
    print(output.shape)