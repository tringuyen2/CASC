import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPVisionConfig

class ImageEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Load CLIP vision model with custom config to force eager attention
        vision_config = CLIPVisionConfig.from_pretrained("openai/clip-vit-base-patch16")
        vision_config.attn_implementation = "eager"  # Force eager attention for compatibility
        
        self.vision_model = CLIPVisionModel.from_pretrained(
            "openai/clip-vit-base-patch16",
            config=vision_config,
            ignore_mismatched_sizes=True
        )
        
        # Projection head for contrastive learning
        self.projection = nn.Linear(768, 256)
        
    def forward(self, images, return_attention=False):
        """
        Forward pass
        Args:
            images: [B, 3, H, W]
            return_attention: whether to return attention scores
        """
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
        if return_attention and outputs.attentions is not None:
            # outputs.attentions is a tuple of (num_layers,)
            # Each element is [B, num_heads, seq_len, seq_len]
            attentions = outputs.attentions[-1]  # Last layer attention [B, H, N+1, N+1]
        
        return {
            'cls': v_cls,
            'local': v_local,
            'projection': self.projection(v_cls),
            'attentions': attentions,
            'hidden_states': outputs.hidden_states
        }