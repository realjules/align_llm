"""Tests for nucleus sampling implementation."""

import numpy as np
import pytest
import torch

from src.sampling.nucleus import NucleusSampling


@pytest.fixture
def sample_logits():
    """Create sample logits for testing."""
    return torch.tensor([
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [5.0, 4.0, 3.0, 2.0, 1.0]
    ])


def test_nucleus_sampling_init():
    """Test nucleus sampler initialization."""
    sampler = NucleusSampling(p=0.9, temperature=0.8)
    assert sampler.p == 0.9
    assert sampler.temperature == 0.8


def test_nucleus_sampling_params():
    """Test getting sampling parameters."""
    sampler = NucleusSampling(p=0.9, temperature=0.8, min_tokens=2)
    params = sampler.get_sampling_params()
    assert params["p"] == 0.9
    assert params["temperature"] == 0.8
    assert params["min_tokens"] == 2


def test_nucleus_sampling(sample_logits):
    """Test basic nucleus sampling functionality."""
    sampler = NucleusSampling(p=0.9)
    result = sampler.sample(sample_logits, num_samples=1)
    
    assert isinstance(result.tokens, torch.Tensor)
    assert isinstance(result.probs, torch.Tensor)
    assert result.tokens.shape == (2, 1)
    assert result.probs.shape == (2, 1)


def test_nucleus_sampling_temperature(sample_logits):
    """Test temperature scaling in nucleus sampling."""
    sampler_high_temp = NucleusSampling(p=0.9, temperature=2.0)
    sampler_low_temp = NucleusSampling(p=0.9, temperature=0.5)
    
    result_high = sampler_high_temp.sample(sample_logits)
    result_low = sampler_low_temp.sample(sample_logits)
    
    # Higher temperature should lead to more uniform probabilities
    assert result_high.entropy > result_low.entropy


def test_nucleus_sampling_min_tokens(sample_logits):
    """Test minimum token constraint."""
    sampler = NucleusSampling(p=0.5, min_tokens=3)
    result = sampler.sample(sample_logits)
    
    # Check that at least min_tokens have non-zero probability
    non_zero_probs = (result.probs > 0).sum()
    assert non_zero_probs >= 3


def test_nucleus_sampling_max_tokens(sample_logits):
    """Test maximum token constraint."""
    sampler = NucleusSampling(p=0.9, max_tokens=2)
    result = sampler.sample(sample_logits)
    
    # Check that no more than max_tokens have non-zero probability
    non_zero_probs = (result.probs > 0).sum()
    assert non_zero_probs <= 2


def test_statistical_guarantee(sample_logits):
    """Test statistical guarantee calculation."""
    sampler = NucleusSampling(p=0.9)
    result = sampler.sample(sample_logits)
    guarantee = sampler.get_statistical_guarantee(result.probs)
    
    assert 0 <= guarantee <= 1
    assert guarantee >= sampler.guarantee_threshold


def test_nucleus_sampling_edge_cases():
    """Test nucleus sampling with edge cases."""
    # Test with uniform logits
    uniform_logits = torch.ones((2, 5))
    sampler = NucleusSampling(p=0.9)
    result = sampler.sample(uniform_logits)
    assert result.entropy > 0.9  # Should be close to maximum entropy
    
    # Test with extreme logits
    extreme_logits = torch.tensor([[100.0, -100.0, -100.0], [-100.0, 100.0, -100.0]])
    result = sampler.sample(extreme_logits)
    assert result.confidence > 0.9  # Should be very confident