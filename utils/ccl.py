import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalContrastiveLearning(nn.Module):
    '''
    Conditional Contrastive Learning (CCL)
    Dynamically adjusts weights based on hard negative similarities
    '''
    def __init__(self, temperature=0.07, lambda_i=2.0, lambda_t=2.0):
        super().__init__()
        self.temperature = temperature
        self.lambda_i = lambda_i
        self.lambda_t = lambda_t
        
    def compute_similarity(self, x, y):
        '''Compute cosine similarity'''
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)
        return torch.matmul(x, y.t())
    
    def contrastive_loss(self, sim_matrix, targets=None):
        '''InfoNCE loss'''
        if targets is None:
            targets = torch.arange(sim_matrix.size(0)).to(sim_matrix.device)
        
        sim_matrix = sim_matrix / self.temperature
        loss = F.cross_entropy(sim_matrix, targets)
        return loss
    
    def forward(self, image_feat, text_feat, caption_feat):
        '''
        Args:
            image_feat: [B, D] image features
            text_feat: [B, D] text features  
            caption_feat: [B, D] caption features
        Returns:
            loss_dict: dictionary of losses
        '''
        # Image-Text Contrastive Learning
        sim_i2t = self.compute_similarity(image_feat, text_feat)
        sim_t2i = sim_i2t.t()
        
        loss_i2t = self.contrastive_loss(sim_i2t)
        loss_t2i = self.contrastive_loss(sim_t2i)
        loss_itc = (loss_i2t + loss_t2i) / 2
        
        # Find hard negatives
        with torch.no_grad():
            # For each sample, find hardest negative
            batch_size = sim_i2t.size(0)
            targets = torch.arange(batch_size).to(sim_i2t.device)
            
            # Mask out positive pairs
            mask = torch.eye(batch_size, device=sim_i2t.device).bool()
            sim_i2t_neg = sim_i2t.clone()
            sim_i2t_neg[mask] = -1e9
            
            sim_t2i_neg = sim_t2i.clone()
            sim_t2i_neg[mask] = -1e9
            
            # Get max similarity with negatives
            S_i2t_n = sim_i2t_neg.max(dim=1)[0]
            S_t2i_n = sim_t2i_neg.max(dim=1)[0]
            S_n = torch.min(S_i2t_n, S_t2i_n)
        
        # Compute caption similarities
        sim_ic = self.compute_similarity(image_feat, caption_feat)
        sim_tc = self.compute_similarity(text_feat, caption_feat)
        
        # Dynamic weights based on hard negatives
        S_ic = torch.diag(sim_ic)
        S_tc = torch.diag(sim_tc)
        
        W_i = torch.where(
            S_ic <= S_n,
            torch.exp((S_ic - S_n) * self.lambda_i),
            torch.ones_like(S_ic)
        )
        
        W_t = torch.where(
            S_tc <= S_n,
            torch.exp((S_tc - S_n) * self.lambda_t),
            torch.ones_like(S_tc)
        )
        
        # Image-Caption Contrastive Learning
        loss_i2c = self.contrastive_loss(sim_ic)
        loss_c2i = self.contrastive_loss(sim_ic.t())
        loss_icc = (loss_i2c + loss_c2i) / 2 * W_i.mean()
        
        # Text-Caption Contrastive Learning  
        loss_t2c = self.contrastive_loss(sim_tc)
        loss_c2t = self.contrastive_loss(sim_tc.t())
        loss_tcc = (loss_t2c + loss_c2t) / 2 * W_t.mean()
        
        return {
            'loss_itc': loss_itc,
            'loss_icc': loss_icc,
            'loss_tcc': loss_tcc,
            'W_i': W_i.mean(),
            'W_t': W_t.mean()
        }