import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class TextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Use pretrained BERT
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Projection head
        self.projection = nn.Linear(768, 256)
        
    def forward(self, text_ids, attention_mask):
        outputs = self.bert(
            input_ids=text_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Get CLS token and sequence outputs
        t_cls = outputs.last_hidden_state[:, 0]  # [B, 768]
        t_seq = outputs.last_hidden_state  # [B, L, 768]
        
        return {
            'cls': t_cls,
            'sequence': t_seq,
            'projection': self.projection(t_cls)
        }
    
    def tokenize(self, texts):
        return self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.config.text_max_length,
            return_tensors='pt'
        )