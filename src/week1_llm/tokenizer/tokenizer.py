from __future__ import annotations

import re
from collections import Counter


class SimpleTokenizer:
    """
    Week 1 educational tokenizer.

    This is intentionally word/punctuation based rather than a production
    GPT-style BPE tokenizer. It demonstrates tokens, IDs, vocabulary,
    special tokens and unknown-token handling.
    """

    SPECIAL_TOKENS = ("<PAD>", "<UNK>", "<BOS>", "<EOS>")

    def __init__(self, vocab_size: int = 5000):
        self.vocab_size = max(vocab_size, len(self.SPECIAL_TOKENS))
        self.word_to_id: dict[str, int] = {}
        self.id_to_word: dict[int, str] = {}
        self._built = False

    @staticmethod
    def split(text: str) -> list[str]:
        return re.findall(r"[A-Za-z0-9_']+|[^\w\s]", text.lower())

    def build_vocab(self, texts: list[str]) -> None:
        counts = Counter(token for text in texts for token in self.split(text))
        self.word_to_id = {
            token: idx for idx, token in enumerate(self.SPECIAL_TOKENS)
        }

        available = self.vocab_size - len(self.SPECIAL_TOKENS)
        for token, _ in counts.most_common(max(0, available)):
            if token not in self.word_to_id:
                self.word_to_id[token] = len(self.word_to_id)

        self.id_to_word = {idx: token for token, idx in self.word_to_id.items()}
        self._built = True

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        if not self._built:
            raise RuntimeError("Call build_vocab() before encode().")

        ids: list[int] = []
        if add_bos:
            ids.append(self.word_to_id["<BOS>"])

        unk = self.word_to_id["<UNK>"]
        ids.extend(self.word_to_id.get(token, unk) for token in self.split(text))

        if add_eos:
            ids.append(self.word_to_id["<EOS>"])
        return ids

    def decode(self, ids: list[int]) -> str:
        if not self._built:
            raise RuntimeError("Call build_vocab() before decode().")
        tokens = [
            self.id_to_word.get(int(idx), "<UNK>")
            for idx in ids
            if self.id_to_word.get(int(idx), "<UNK>") not in {"<PAD>", "<BOS>", "<EOS>"}
        ]
        text = " ".join(tokens)
        text = re.sub(r"\s+([,.!?;:])", r"\1", text)
        return text

    @property
    def vocab_size_actual(self) -> int:
        return len(self.word_to_id)
