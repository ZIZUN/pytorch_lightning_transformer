
import torch.nn as nn
import torch

import pytorch_lightning as pl

from util.others.my_metrics import Accuracy
from util.others.dist_utils import is_main_process

from .bert_modeling import BertConfig, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput


from transformers import (
    get_cosine_schedule_with_warmup
)

class IntentCLSModule(pl.LightningModule):
    def __init__(self, _config, num_labels=2):
        super().__init__()
        self.save_hyperparameters()
        
        model_config = BertConfig()
        model_config.bert_hidden_size = _config['bert_hidden_size']
        model_config.bert_layer_num = _config['bert_layer_num']
        model_config.multi_head_num = _config['multi_head_num']
        model_config.qkv_hidden_size = _config['qkv_hidden_size']
        model_config.vocab_size = _config['vocab_size']
        
        self.model = BertModel(config=model_config)
        
        self.classifier = nn.Linear(model_config.bert_hidden_size, num_labels)
            
        self.metric = Accuracy()

    def forward(self, input_ids, attention_mask, labels=None):
        
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        logits = self.classifier(outputs[:,0,:].squeeze(1))
        
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )
        
    def training_step(self, batch, batch_idx):
        output = self(**batch)
        
        self.log(f"train/loss", output.loss)
        return output.loss
    
    def validation_step(self, batch, batch_idx):
        output = self(**batch)
        self.metric.update(output.logits, batch['labels'])
        
        self.log(f"val/loss", output.loss)

    def validation_epoch_end(self, outs):
        accuracy = self.metric.compute().tolist()
        # if is_main_process():
        #     print(f'accuracy: {str(accuracy)}')
        self.metric.reset()
        
        self.log(f"val/accuracy", accuracy)
        
    def test_step(self, batch, batch_idx):
        output = self(**batch)
        self.metric.update(output.logits, batch['labels'])
    
    def configure_optimizers(self):
        param_optimizer = self.named_parameters()
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [{
            'params': [
                p for n, p in param_optimizer
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay':
            0.01
        }, {
            'params':
            [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay':
            0.0
        }]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=0.00005, betas=(0.9, 0.999))
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams._config['warmup_steps'], num_training_steps=self.hparams._config['max_steps']
        )

        sched = {"scheduler": scheduler, "interval": "step"}

        return (
            [optimizer],
            [sched],
        )
