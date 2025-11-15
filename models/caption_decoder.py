import torch
import torch.nn as nn
from transformers import BertLMHeadModel

class CaptionDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Use BERT for language modeling
        self.lm_model = BertLMHeadModel.from_pretrained('bert-base-uncased')
        
        # Cross-attention layer
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=768,
            num_heads=12,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(768)
        
    def forward(self, text_embeddings, image_embeddings, text_ids=None):
        # Cross attention: text queries image
        cross_out, _ = self.cross_attention(
            query=text_embeddings,
            key=image_embeddings,
            value=image_embeddings
        )
        
        # Residual connection and layer norm
        fused = self.layer_norm(text_embeddings + cross_out)
        
        # Generate caption
        if text_ids is not None:
            outputs = self.lm_model(
                inputs_embeds=fused,
                labels=text_ids
            )
            return outputs.loss, fused
        else:
            # Inference mode
            return None, fused