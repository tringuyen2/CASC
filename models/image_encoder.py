import torch
import torch.nn as nn
from transformers import CLIPVisionModel

class ImageEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Use pretrained CLIP vision encoder
        self.vision_model = CLIPVisionModel.from_pretrained(
            "openai/clip-vit-base-patch16"
        )
        
        # Projection head for contrastive learning
        self.projection = nn.Linear(768, 256)
        
    def forward(self, images, return_attention=False):
        outputs = self.vision_model(
            images,
            output_attentions=return_attention,
            output_hidden_states=True
        )
        
        # Get class token (CLS) and local tokens
        last_hidden_state = outputs.last_hidden_state
        v_cls = last_hidden_state[:, 0]  # [B, 768]
        v_local = last_hidden_state[:, 1:]  # [B, N, 768]
        
        # Get attention scores for GAS
        attentions = None
        if return_attention:
            attentions = outputs.attentions[-1]  # Last layer attention
        
        return {
            'cls': v_cls,
            'local': v_local,
            'projection': self.projection(v_cls),
            'attentions': attentions
        }