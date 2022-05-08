from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.roberta.modeling_roberta import RobertaConfig, RobertaForSequenceClassification
import torch.nn as nn
import torch

import pytorch_lightning as pl

from util.others.my_metrics import Accuracy
from util.others.dist_utils import is_main_process

from transformers import (
    get_cosine_schedule_with_warmup
)

class Intent_CLS_Module(pl.LightningModule):
    def __init__(self, _config, num_labels=2):
        super().__init__()
        self.save_hyperparameters()
                
        if _config['model_name'] == 'roberta-base':
            model_config = RobertaConfig.from_pretrained(pretrained_model_name_or_path="roberta-base",
                                                         hidden_dropout_prob=0.1, num_labels=num_labels)
            self.model = RobertaForSequenceClassification.from_pretrained("roberta-base", config=model_config)
            
        self.metric = Accuracy()

    def forward(self, input_ids, attention_mask, labels=None):
        
        outputs = self.model.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        logits = self.model.classifier(sequence_output)
        
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
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
