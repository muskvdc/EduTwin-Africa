
from __future__ import annotations

import torch

from week1_llm.tokenizer.tokenizer import SimpleTokenizer
from week1_llm.transformer.model import SmallCausalLanguageModel


@torch.no_grad()
def generate_text(
    model: SmallCausalLanguageModel,
    tokenizer: SimpleTokenizer,
    prompt: str,
    max_new_tokens: int = 20,
    temperature: float = 0.8,
    do_sample: bool = True,
) -> str:
    model.eval()
    token_ids = tokenizer.encode(prompt)
    if not token_ids:
        raise ValueError("Prompt produced no tokens.")
    input_ids = torch.tensor([token_ids], dtype=torch.long)
    generated = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
    )
    return tokenizer.decode(generated[0].tolist())
