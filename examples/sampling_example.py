"""Example usage of sampling methods."""

import torch
from src.sampling.nucleus import NucleusSampling
from src.sampling.speculative import SpeculativeDecoder


def main():
    # Create sample logits
    logits = torch.tensor([
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [5.0, 4.0, 3.0, 2.0, 1.0]
    ])
    
    # Initialize nucleus sampler
    nucleus_sampler = NucleusSampling(
        p=0.9,
        temperature=0.8,
        min_tokens=2,
        guarantee_threshold=0.95
    )
    
    # Sample using nucleus sampling
    print("Nucleus Sampling Example:")
    result = nucleus_sampler.sample(logits, num_samples=3)
    print(f"Sampled tokens: {result.tokens}")
    print(f"Probabilities: {result.probs}")
    print(f"Entropy: {result.entropy}")
    print(f"Confidence: {result.confidence}")
    print(f"Statistical guarantee: {nucleus_sampler.get_statistical_guarantee(result.probs)}\n")
    
    # Create dummy models for speculative decoding example
    class DummyModel(torch.nn.Module):
        def forward(self, x):
            return x * 2
    
    draft_model = DummyModel()
    target_model = DummyModel()
    
    # Initialize speculative decoder
    spec_decoder = SpeculativeDecoder(
        draft_model=draft_model,
        target_model=target_model,
        max_steps=4,
        temperature=0.8,
        top_k=50,
        top_p=0.9
    )
    
    # Sample using speculative decoding
    print("Speculative Decoding Example:")
    result = spec_decoder.sample(logits, num_samples=3)
    print(f"Sampled tokens: {result.tokens}")
    print(f"Probabilities: {result.probs}")
    print(f"Entropy: {result.entropy}")
    print(f"Confidence: {result.confidence}")


if __name__ == "__main__":
    main()