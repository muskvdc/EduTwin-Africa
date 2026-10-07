import torch

from week1_llm.transformer.model import SmallCausalLanguageModel


def test_language_model_shapes():
    model = SmallCausalLanguageModel(
        vocab_size=20,
        embedding_dim=16,
        context_length=8,
        hidden_dim=32,
    )
    ids = torch.randint(0, 20, (2, 8))
    logits, loss, weights = model(ids, ids)
    assert logits.shape == (2, 8, 20)
    assert loss is not None
    assert weights.shape == (2, 8, 8)
