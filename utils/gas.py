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
        
    def compute_attention_from_features(self, cls_token, local_features):
        """
        Compute attention scores from features when attention maps are not available
        Args:
            cls_token: [B, D]
            local_features: [B, N, D]
        Returns:
            attention_scores: [B, N]
        """
        # Normalize features
        cls_norm = torch.nn.functional.normalize(cls_token, dim=-1)
        local_norm = torch.nn.functional.normalize(local_features, dim=-1)
        
        # Compute similarity as attention [B, N]
        attention_scores = torch.bmm(
            cls_norm.unsqueeze(1),  # [B, 1, D]
            local_norm.transpose(1, 2)  # [B, D, N]
        ).squeeze(1)  # [B, N]
        
        return attention_scores
        
    def forward(self, attention_scores, local_features, cls_token=None):
        '''
        Args:
            attention_scores: [B, H, N+1, N+1] attention from transformer or None
            local_features: [B, N, D] local patch features
            cls_token: [B, D] class token (used if attention_scores is None)
        Returns:
            mask: [B, N] binary mask
            masked_features: [B, K', D] selected features
        '''
        B, N, D = local_features.shape
        
        # Handle case when attention_scores is None
        if attention_scores is None:
            if cls_token is None:
                # If no attention and no cls token, use all features
                print("Warning: No attention scores and no cls token, using all features")
                mask = torch.ones(B, N, device=local_features.device)
                return mask, local_features
            else:
                # Compute attention from features
                A_cls = self.compute_attention_from_features(cls_token, local_features)
        else:
            # Step 1: Compute CLS token similarity with local tokens
            # Average across all heads
            A_cls = attention_scores.mean(dim=1)  # [B, N+1, N+1]
            A_cls = A_cls[:, 0, 1:]  # [B, N] - CLS to patches
        
        # Ensure top_k doesn't exceed number of tokens
        actual_k = min(self.top_k, N)
        
        # Select Top-K tokens
        topk_values, topk_indices = torch.topk(A_cls, actual_k, dim=1)
        
        # Initialize mask
        mask = torch.zeros(B, N, device=local_features.device)
        mask.scatter_(1, topk_indices, 1.0)
        
        # Step 2: For each top-K token, find its most similar neighbor
        for i in range(actual_k):
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
            
            # Check if there are any unselected tokens
            if similarity.max() > 0:
                max_idx = similarity.argmax(dim=1, keepdim=True)  # [B, 1]
                mask.scatter_(1, max_idx, 1.0)
        
        # Apply mask to get selected features
        mask_expanded = mask.unsqueeze(-1)  # [B, N, 1]
        masked_features = local_features * mask_expanded
        
        # Remove zero vectors and reshape
        num_selected = mask.sum(dim=1).max().int().item()
        if num_selected == 0:
            num_selected = N  # Use all if none selected
        
        # Get top features based on mask
        result = []
        for b in range(B):
            selected = masked_features[b][mask[b] > 0]
            if len(selected) < num_selected:
                # Pad if needed
                padding = torch.zeros(num_selected - len(selected), D, device=local_features.device)
                selected = torch.cat([selected, padding], dim=0)
            result.append(selected[:num_selected])
        
        masked_features = torch.stack(result, dim=0)  # [B, K', D]
        
        return mask, masked_features