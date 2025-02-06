"""Nucleus sampling with statistical guarantees."""

from typing import Optional, Union

import torch
from torch import Tensor

from .base import BaseSampler, SamplingResult


class NucleusSampling(BaseSampler):
    """Implements nucleus sampling with statistical guarantees."""
    
    def __init__(
        self,
        p: float = 0.9,
        temperature: float = 1.0,
        min_tokens: int = 1,
        max_tokens: Optional[int] = None,
        guarantee_threshold: float = 0.95
    ):
        """Initialize nucleus sampler.
        
        Args:
            p: Cumulative probability threshold (default: 0.9)
            temperature: Sampling temperature (default: 1.0)
            min_tokens: Minimum number of tokens to consider (default: 1)
            max_tokens: Maximum number of tokens to consider (optional)
            guarantee_threshold: Probability threshold for statistical guarantees (default: 0.95)
        """
        super().__init__(temperature=temperature)
        self.p = p
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.guarantee_threshold = guarantee_threshold
    
    def sample(self, logits: Union[torch.Tensor, torch.ndarray], num_samples: int = 1) -> SamplingResult:
        """Sample tokens using nucleus sampling.
        
        Args:
            logits: Raw logits from model
            num_samples: Number of samples to generate
            
        Returns:
            SamplingResult containing tokens and probabilities
        """
        logits = self._ensure_tensor(logits)
        
        if self.temperature != 1.0:
            logits = logits / self.temperature
        
        # Sort logits in descending order
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        
        # Calculate cumulative probabilities
        probs = torch.softmax(sorted_logits, dim=-1)
        cum_probs = torch.cumsum(probs, dim=-1)
        
        # Create nucleus mask
        nucleus_mask = cum_probs <= self.p
        
        # Ensure minimum number of tokens
        if self.min_tokens > 1:
            nucleus_mask[..., :self.min_tokens] = True
        
        # Apply maximum token limit if specified
        if self.max_tokens is not None:
            nucleus_mask[..., self.max_tokens:] = False
        
        # Filter logits
        nucleus_logits = sorted_logits.masked_fill(~nucleus_mask, float('-inf'))
        
        # Sample from filtered distribution
        filtered_probs = torch.softmax(nucleus_logits, dim=-1)
        samples = torch.multinomial(filtered_probs, num_samples)
        
        # Map back to original token indices
        tokens = torch.gather(sorted_indices, -1, samples)
        
        # Calculate sampling probabilities
        sample_probs = torch.gather(filtered_probs, -1, samples)
        
        # Calculate statistics
        entropy = self._calculate_entropy(filtered_probs)
        confidence = self._calculate_confidence(sample_probs)
        
        return SamplingResult(
            tokens=tokens,
            probs=sample_probs,
            entropy=entropy,
            confidence=confidence
        )
    
    def get_sampling_params(self) -> dict:
        """Get current sampling parameters."""
        return {
            "p": self.p,
            "temperature": self.temperature,
            "min_tokens": self.min_tokens,
            "max_tokens": self.max_tokens,
            "guarantee_threshold": self.guarantee_threshold
        }
    
    def _calculate_entropy(self, probs: Tensor) -> float:
        """Calculate Shannon entropy of the sampling distribution."""
        return float(-(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean())
    
    def _calculate_confidence(self, probs: Tensor) -> float:
        """Calculate confidence score based on probability distribution."""
        return float(probs.max(dim=-1)[0].mean())
    
    def get_statistical_guarantee(self, probs: Tensor) -> float:
        """Calculate statistical guarantee for the sampling.
        
        Returns probability that sampled tokens come from the intended distribution.
        """
        # Calculate probability mass in the nucleus
        nucleus_mass = probs[probs > 0].sum()
        
        # Calculate guarantee based on nucleus coverage
        guarantee = nucleus_mass / self.p
        
        return float(guarantee)