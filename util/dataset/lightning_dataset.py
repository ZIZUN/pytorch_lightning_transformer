import functools

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset
import torch

from transformers import AutoTokenizer
from tqdm import tqdm


class Intent_CLS_DataModule(LightningDataModule):
    def __init__(self, _config, dist=True):
        super().__init__()
        
        self.per_gpu_batch_size = _config['per_gpu_batch_size']
        self.num_workers = _config['num_workers']
        self.input_seq_len = _config['input_seq_len']
        self.dist = dist
        
        self.train_dataset_path = _config['train_dataset_path']
        self.val_dataset_path = _config['val_dataset_path']
        self.test_dataset_path = _config['test_dataset_path']
        self.model_name = _config['model_name']
        
        _, self.train_labels_li = load_intent_examples(self.train_dataset_path)
        _, self.val_labels_li = load_intent_examples(self.val_dataset_path)
        _, self.test_labels_li = load_intent_examples(self.test_dataset_path)

    # def prepare_data(self):
    #     NotImplemented
        
    def setup(self, stage):
        self.train_dataset = LoadDataset(self.model_name, self.train_dataset_path, self.train_labels_li, seq_len=self.input_seq_len)        
        self.val_dataset = LoadDataset(self.model_name, self.val_dataset_path, self.val_labels_li, seq_len=self.input_seq_len)
        self.test_dataset = LoadDataset(self.model_name, self.test_dataset_path, self.test_labels_li, seq_len=self.input_seq_len)
        
        self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True)
        self.val_sampler = DistributedSampler(self.val_dataset, shuffle=True)
        self.test_sampler = DistributedSampler(self.test_dataset, shuffle=False)

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.per_gpu_batch_size,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
        )
        return loader

    def val_dataloader(self, batch_size=None):
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.per_gpu_batch_size,
            sampler=self.val_sampler,
            num_workers=self.num_workers,
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.per_gpu_batch_size,
            sampler=self.test_sampler,
            num_workers=self.num_workers,
        )
        return loader

    
    



class IntentExample:
    def __init__(self, text, label, do_lower_case):
        self.original_text = text
        self.text = text
        self.label = label

        if do_lower_case:
            self.text = self.text.lower()



def load_intent_examples(file_path, do_lower_case=True):
    examples = []
    
    labels_li = []
    with open('{}/seq.in'.format(file_path), 'r', encoding="utf-8") as f_text, open('{}/label'.format(file_path), 'r', encoding="utf-8") as f_label:
        for text, label in zip(f_text, f_label):
            text = text.strip()
            label = label.strip()
            
            if label not in labels_li:
                labels_li.append(label)
            
            e = IntentExample(text, label, do_lower_case)
            examples.append(e)
    return examples, labels_li
    
    
class LoadDataset(Dataset):
    def __init__(self, model_name, corpus_path, labels_li, seq_len):
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        self.seq_len = seq_len
        self.corpus_path = corpus_path
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.start = self.tokenizer.bos_token_id
        self.sep = self.tokenizer.eos_token_id
        self.padding = self.tokenizer.pad_token_id

        self.dataset, _ = load_intent_examples(file_path=corpus_path)
        self.labels_li = labels_li
        self.dataset_len = len(self.dataset)
        
        self.processed_dataset = []

        for data in tqdm(self.dataset):
            text = data.text
            label = data.label
            label = self.labels_li.index(label)

            text = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))

            if len(text) <= self.seq_len - 2:
                text = [self.start] + text + [self.sep]
                pad_length = self.seq_len - len(text)

                attention_mask = (len(text) * [1]) + (pad_length * [0])
                text = text + (pad_length * [self.padding])
            else:
                text = text[:self.seq_len - 2]
                text = [self.start] + text + [self.sep]
                attention_mask = len(text) * [1]

            model_input = text
            model_label = int(label)
            
            self.processed_dataset.append({"input_ids": model_input, 'attention_mask': attention_mask, "labels": model_label})

    def __len__(self):
        return len(self.processed_dataset)

    def __getitem__(self, item):
        output = self.processed_dataset[item]
        return {key: torch.tensor(value) for key, value in output.items()}