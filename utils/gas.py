import torch
import torch.nn as nn

class GranularityAwarenessSensor(nn.Module):
    '''
    Granularity Awareness Sensor (GAS)
    Adaptive masking strategy for fine-grained feature selection
    '''
    def __init__(self, top_k=45):
        super().__init__()
        self.top_k = top_k
        
    def forward(self, attention_scores, local_features):
        '''
        Args:
            attention_scores: [B, H, N+1, N+1] attention from last transformer layer
            local_features: [B, N, D] local patch features
        Returns:
            mask: [B, N] binary mask
            masked_features: [B, K', D] selected features
        '''
        B, N, D = local_features.shape
        
        # Step 1: Compute CLS token similarity with local tokens
        # Average across all heads
        A_cls = attention_scores.mean(dim=1)  # [B, N+1, N+1]
        A_cls = A_cls[:, 0, 1:]  # [B, N] - CLS to patches
        
        # Select Top-K tokens
        topk_values, topk_indices = torch.topk(A_cls, self.top_k, dim=1)
        
        # Initialize mask
        mask = torch.zeros(B, N, device=local_features.device)
        mask.scatter_(1, topk_indices, 1.0)
        
        # Step 2: For each top-K token, find its most similar neighbor
        for i in range(self.top_k):
            # Get query tokens
            query_idx = topk_indices[:, i:i+1]  # [B, 1]
            query_features = torch.gather(
                local_features, 
                1, 
                query_idx.unsqueeze(-1).expand(-1, -1, D)
            )  # [B, 1, D]
            
            # Compute similarity with all local tokens
            similarity = torch.bmm(
                query_features, 
                local_features.transpose(1, 2)
            )  # [B, 1, N]
            similarity = similarity.squeeze(1)  # [B, N]
            
            # Find most similar token (excluding already selected)
            similarity = similarity * (1 - mask)  # Mask out selected tokens
            max_idx = similarity.argmax(dim=1, keepdim=True)  # [B, 1]
            mask.scatter_(1, max_idx, 1.0)
        
        # Apply mask to get selected features
        mask_expanded = mask.unsqueeze(-1)  # [B, N, 1]
        masked_features = local_features * mask_expanded
        
        # Remove zero vectors
        masked_features = masked_features[mask_expanded.squeeze(-1) > 0].view(
            B, -1, D
        )
        
        return mask, masked_features