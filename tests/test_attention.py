import torch

from week1_llm.transformer.attention import SelfAttention


def test_attention_is_causal():
    torch.manual_seed(42)
    attention = SelfAttention(8)
    x = torch.randn(1, 4, 8)
    _, weights = attention(x)

    # Future positions must receive zero attention.
    upper = torch.triu(weights[0], diagonal=1)
    assert torch.allclose(upper, torch.zeros_like(upper))
