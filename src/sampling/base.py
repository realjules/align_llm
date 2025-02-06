"""Base classes for sampling methods."""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor


class BaseSampler(ABC):
    """Base class for all sampling methods."""
    
    def __init__(self, temperature: float = 1.0):
        """Initialize the sampler.
        
        Args:
            temperature: Sampling temperature (default: 1.0)
        """
        self.temperature = temperature
    
    @abstractmethod
    def sample(self, logits: Union[np.ndarray, Tensor], num_samples: int = 1) -> Tuple[Tensor, Tensor]:
        """Sample from logits.
        
        Args:
            logits: Raw logits from model (shape: batch_size x vocab_size)
            num_samples: Number of samples to generate
            
        Returns:
            Tuple of:
                - Sampled token indices (shape: batch_size x num_samples)
                - Sampling probabilities (shape: batch_size x num_samples)
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_sampling_params(self) -> dict:
        """Get the current sampling parameters.
        
        Returns:
            Dictionary of sampling parameters
        """
        raise NotImplementedError
    
    def _ensure_tensor(self, x: Union[np.ndarray, Tensor]) -> Tensor:
        """Convert input to PyTorch tensor if needed."""
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        return x


class SamplingResult:
    """Container for sampling results with statistical information."""
    
    def __init__(
        self,
        tokens: Tensor,
        probs: Tensor,
        entropy: Optional[float] = None,
        confidence: Optional[float] = None
    ):
        """Initialize sampling result.
        
        Args:
            tokens: Sampled token indices
            probs: Sampling probabilities
            entropy: Shannon entropy of the sampling distribution (optional)
            confidence: Confidence score for the samples (optional)
        """
        self.tokens = tokens
        self.probs = probs
        self.entropy = entropy
        self.confidence = confidence
    
    def get_stats(self) -> dict:
        """Get statistical information about the sampling.
        
        Returns:
            Dictionary containing sampling statistics
        """
        stats = {
            "mean_prob": float(self.probs.mean()),
            "min_prob": float(self.probs.min()),
            "max_prob": float(self.probs.max()),
        }
        if self.entropy is not None:
            stats["entropy"] = self.entropy
        if self.confidence is not None:
            stats["confidence"] = self.confidence
        return stats