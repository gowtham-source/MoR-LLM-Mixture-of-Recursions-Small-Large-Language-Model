"""
Dynamic routing mechanisms for MoR-SLM.

Implements Expert-Choice and Token-Choice routing strategies as described in the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class BaseRouter(nn.Module):
    """Base class for routing mechanisms."""
    
    def __init__(self, hidden_size: int, router_type: str = "linear"):
        super().__init__()
        self.hidden_size = hidden_size
        self.router_type = router_type
        
        if router_type == "linear":
            self.router = nn.Linear(hidden_size, 1, bias=False)
        elif router_type == "mlp":
            self.router = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 4),
                nn.ReLU(),
                nn.Linear(hidden_size // 4, 1)
            )
        else:
            raise ValueError(f"Unknown router type: {router_type}")
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute routing scores."""
        return self.router(hidden_states)


class ExpertChoiceRouter(BaseRouter):
    """
    Expert-Choice routing mechanism.
    
    Implements hierarchical token selection where only previously selected
    tokens can continue to deeper recursion levels.
    """
    
    def __init__(self, hidden_size: int, router_type: str = "linear", 
                 beta_percentile: float = 0.8, auxiliary_loss_weight: float = 0.01):
        super().__init__(hidden_size, router_type)
        self.beta_percentile = beta_percentile
        self.auxiliary_loss_weight = auxiliary_loss_weight
        
    def compute_threshold(self, scores: torch.Tensor) -> torch.Tensor:
        """Compute β-percentile threshold P_β(G^r)."""
        batch_size, seq_len = scores.shape
        # Flatten scores for percentile computation
        flat_scores = scores.view(-1)
        k = int(self.beta_percentile * flat_scores.numel())
        threshold, _ = torch.kthvalue(flat_scores, k)
        return threshold
    
    def forward(self, hidden_states: torch.Tensor, recursion_step: int,
                previous_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for expert-choice routing.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            recursion_step: Current recursion step (0-indexed)
            previous_mask: Mask from previous recursion step
            
        Returns:
            routing_scores: Raw routing scores
            selection_mask: Binary mask for selected tokens
            auxiliary_loss: Auxiliary loss for causality violation mitigation
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Compute routing scores: g_t^r = G(θ_r^T * H_t^r)
        raw_scores = self.router(hidden_states).squeeze(-1)  # [batch_size, seq_len]
        routing_scores = torch.sigmoid(raw_scores)
        
        # Apply β-percentile threshold
        threshold = self.compute_threshold(routing_scores)
        
        # Create selection mask
        selection_mask = (routing_scores >= threshold).float()
        
        # Apply hierarchical constraint: only previously selected tokens can continue
        if previous_mask is not None:
            selection_mask = selection_mask * previous_mask
        
        # Compute auxiliary loss for causality violation mitigation
        # This encourages the router to respect causal dependencies
        auxiliary_loss = self._compute_auxiliary_loss(routing_scores, selection_mask)
        
        return routing_scores, selection_mask, auxiliary_loss
    
    def _compute_auxiliary_loss(self, scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute auxiliary loss to mitigate causality violations."""
        # Simple auxiliary loss that encourages consistent routing patterns
        batch_size, seq_len = scores.shape
        
        # Loss 1: Encourage smooth transitions in selection patterns
        if seq_len > 1:
            mask_diff = torch.diff(mask, dim=-1)  # Differences between adjacent tokens
            consistency_loss = torch.abs(mask_diff).mean()  # Penalty for abrupt changes
        else:
            consistency_loss = torch.tensor(0.0, device=mask.device)
        
        # Loss 2: Encourage some tokens to be selected (avoid all zeros)
        selection_rate = mask.mean()
        sparsity_loss = F.relu(0.1 - selection_rate)  # Encourage at least 10% selection
        
        # Loss 3: Prevent over-selection (avoid all ones)
        over_selection_loss = F.relu(selection_rate - 0.9)  # Discourage more than 90% selection
        
        auxiliary_loss = (consistency_loss + sparsity_loss + over_selection_loss) * self.auxiliary_loss_weight
        
        return auxiliary_loss


class TokenChoiceRouter(BaseRouter):
    """
    Token-Choice routing mechanism.
    
    Alternative routing strategy where tokens choose their recursion depth.
    """
    
    def __init__(self, hidden_size: int, max_recursion_depth: int, 
                 router_type: str = "linear"):
        super().__init__(hidden_size, router_type)
        self.max_recursion_depth = max_recursion_depth
        
        # Router outputs probability distribution over recursion depths
        if router_type == "linear":
            self.router = nn.Linear(hidden_size, max_recursion_depth)
        elif router_type == "mlp":
            self.router = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 4),
                nn.ReLU(),
                nn.Linear(hidden_size // 4, max_recursion_depth)
            )
    
    def forward(self, hidden_states: torch.Tensor, 
                current_depth: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for token-choice routing.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            current_depth: Current recursion depth
            
        Returns:
            depth_probs: Probability distribution over recursion depths
            continue_mask: Binary mask indicating which tokens continue
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Compute depth probabilities
        depth_logits = self.router(hidden_states)  # [batch_size, seq_len, max_depth]
        depth_probs = F.softmax(depth_logits, dim=-1)
        
        # Determine which tokens continue based on their chosen depth
        # Tokens continue if their chosen depth > current_depth
        continue_probs = depth_probs[:, :, current_depth+1:].sum(dim=-1)
        continue_mask = (continue_probs > 0.5).float()
        
        return depth_probs, continue_mask


class RoutingLoss(nn.Module):
    """Combined loss function for routing mechanisms."""
    
    def __init__(self, auxiliary_weight: float = 0.01):
        super().__init__()
        self.auxiliary_weight = auxiliary_weight
    
    def forward(self, auxiliary_losses: list, main_loss: torch.Tensor) -> torch.Tensor:
        """Combine main loss with auxiliary routing losses."""
        total_auxiliary = sum(auxiliary_losses) if auxiliary_losses else 0
        return main_loss + self.auxiliary_weight * total_auxiliary
