import torch
import torch.nn as nn
from models.image_encoder import ImageEncoder
from models.text_encoder import TextEncoder
from models.caption_decoder import CaptionDecoder
from utils.gas import GranularityAwarenessSensor
from utils.ccl import ConditionalContrastiveLearning

class CASCModel(nn.Module):
    '''
    CASC: Cross-modal Alignment with Synthetic Caption
    Main model for Text-based Person Search
    '''
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Encoders and Decoder
        self.image_encoder = ImageEncoder(config)
        self.text_encoder = TextEncoder(config)
        self.caption_decoder = CaptionDecoder(config)
        
        # GAS and CCL modules
        self.gas = GranularityAwarenessSensor(top_k=config.top_k)
        self.ccl = ConditionalContrastiveLearning(
            temperature=config.temperature,
            lambda_i=config.lambda_i,
            lambda_t=config.lambda_t
        )
        
        # Momentum encoders
        self.image_encoder_m = ImageEncoder(config)
        self.text_encoder_m = TextEncoder(config)
        
        self._init_momentum_models()
        
        # Cross encoder for ITM
        self.cross_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=768, nhead=12),
            num_layers=6
        )
        self.itm_head = nn.Linear(768, 2)
        
    def _init_momentum_models(self):
        '''Initialize momentum models'''
        for param, param_m in zip(
            self.image_encoder.parameters(),
            self.image_encoder_m.parameters()
        ):
            param_m.data.copy_(param.data)
            param_m.requires_grad = False
            
        for param, param_m in zip(
            self.text_encoder.parameters(),
            self.text_encoder_m.parameters()
        ):
            param_m.data.copy_(param.data)
            param_m.requires_grad = False
    
    @torch.no_grad()
    def _momentum_update(self):
        '''Update momentum models'''
        for param, param_m in zip(
            self.image_encoder.parameters(),
            self.image_encoder_m.parameters()
        ):
            param_m.data = param_m.data * self.config.momentum + \
                          param.data * (1 - self.config.momentum)
                          
        for param, param_m in zip(
            self.text_encoder.parameters(),
            self.text_encoder_m.parameters()
        ):
            param_m.data = param_m.data * self.config.momentum + \
                          param.data * (1 - self.config.momentum)
    
    def forward(self, images, text_ids, attention_mask, labels=None):
        '''
        Forward pass
        Args:
            images: [B, 3, H, W]
            text_ids: [B, L]
            attention_mask: [B, L]
            labels: [B, L] for language modeling (optional)
        '''
        # Encode image with attention for GAS
        img_out = self.image_encoder(images, return_attention=True)
        v_cls = img_out['cls']
        v_local = img_out['local']
        v_proj = img_out['projection']
        attentions = img_out['attentions']
        
        # Encode text
        txt_out = self.text_encoder(text_ids, attention_mask)
        t_cls = txt_out['cls']
        t_seq = txt_out['sequence']
        t_proj = txt_out['projection']
        
        # Apply GAS to select discriminative features
        mask, v_masked = self.gas(attentions, v_local)
        
        # Generate caption with masked image features
        loss_lm, caption_feat = self.caption_decoder(
            t_seq, v_masked, labels
        )
        
        # Get caption projection
        c_proj = caption_feat[:, 0]  # Use first token as caption feature
        
        # Conditional Contrastive Learning
        ccl_losses = self.ccl(v_proj, t_proj, c_proj)
        
        # Image-Text Matching (simplified)
        # Concatenate image and text features
        combined = torch.cat([v_cls.unsqueeze(1), t_seq], dim=1)
        cross_out = self.cross_encoder(combined.transpose(0, 1))
        itm_logits = self.itm_head(cross_out[0])
        
        # Create ITM labels (1 for positive pairs, 0 for negatives)
        batch_size = images.size(0)
        itm_labels = torch.arange(batch_size).to(images.device)
        itm_labels = (itm_labels == itm_labels.view(-1, 1)).long()
        loss_itm = nn.CrossEntropyLoss()(itm_logits, itm_labels[:, 0])
        
        # Total loss
        total_loss = (
            self.config.weight_itc * ccl_losses['loss_itc'] +
            self.config.weight_icc * ccl_losses['loss_icc'] +
            self.config.weight_tcc * ccl_losses['loss_tcc'] +
            self.config.weight_itm * loss_itm +
            self.config.weight_lm * loss_lm
        )
        
        # Update momentum models
        self._momentum_update()
        
        return {
            'total_loss': total_loss,
            'loss_itc': ccl_losses['loss_itc'],
            'loss_icc': ccl_losses['loss_icc'],
            'loss_tcc': ccl_losses['loss_tcc'],
            'loss_itm': loss_itm,
            'loss_lm': loss_lm,
            'W_i': ccl_losses['W_i'],
            'W_t': ccl_losses['W_t']
        }
    
    @torch.no_grad()
    def extract_features(self, images=None, text_ids=None, attention_mask=None):
        '''Extract features for inference'''
        if images is not None:
            img_out = self.image_encoder(images)
            return img_out['projection']
        elif text_ids is not None:
            txt_out = self.text_encoder(text_ids, attention_mask)
            return txt_out['projection']
        else:
            raise ValueError("Either images or text_ids must be provided")