from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from week1_llm.api.groq_client import GroqClient
from week1_llm.embeddings.embeddings import WordEmbeddings
from week1_llm.foundations.neural_network import TwoLayerNeuralNetwork
from week1_llm.digital_twin.predictor import DigitalTwinPredictor
from week1_llm.digital_twin.data import ACTIVITY_NAMES, generate_synthetic_behavioral_data
from week1_llm.foundations.probability import MarkovActivityPredictor, distribution
from week1_llm.prompting.prompts import PromptBuilder
from week1_llm.tokenizer.tokenizer import SimpleTokenizer
from week1_llm.transformer.attention import SelfAttention
from week1_llm.transformer.model import SmallCausalLanguageModel




def demo_probability() -> None:
    sequence = [0, 1, 1, 1, 1, 1, 2, 0, 1, 1, 1, 1, 1, 4]
    print("\n[1] Probability")
    for activity, p in distribution(sequence).items():
        print(f"  {ACTIVITY_NAMES.get(activity, activity)}: {p:.2%}")

    predictor = MarkovActivityPredictor().fit(sequence)
    print("  P(next | Work):", {
        ACTIVITY_NAMES.get(k, k): f"{v:.2%}"
        for k, v in predictor.predict_distribution(1).items()
    })


def demo_neural_network() -> None:
    print("\n[2] Neural network / backpropagation")
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 1], dtype=float)
    model = TwoLayerNeuralNetwork(input_size=2, hidden_size=4, seed=42)
    model.fit(X, y, epochs=1000, learning_rate=0.5)
    print("  predictions:", model.predict(X).tolist())
    print(f"  final loss: {model.loss_history[-1]:.4f}")


def demo_digital_twin() -> None:
    print("\n[3] Digital Twin")
    X, y = generate_synthetic_behavioral_data(n_samples=1000, seed=42)

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    twin = DigitalTwinPredictor(
        input_size=5,
        num_activities=5,
        hidden_size=32,
    )
    twin.fit(X_train, y_train, epochs=2000, learning_rate=0.08)

    train_accuracy = float(np.mean(twin.predict(X_train) == y_train))
    test_accuracy = float(np.mean(twin.predict(X_test) == y_test))

    probabilities = twin.predict_probabilities(X_test[:1])[0]
    prediction = int(np.argmax(probabilities))
    print("  samples:", len(X))
    print("  split:", f"{len(X_train)}/{len(X_test)} (80/20)")
    print("  train accuracy:", f"{train_accuracy:.1%}")
    print("  test accuracy:", f"{test_accuracy:.1%}")
    print("  example prediction:", ACTIVITY_NAMES[prediction])
    print("  probability vector:", np.round(probabilities, 3).tolist())

def demo_tokenizer_embeddings() -> None:
    print("\n[3] Tokenization + embeddings")
    texts = [
        "I enjoy running as a healthy hobby.",
        "My passion is faith and coding.",
        "Coding helps me build useful AI systems.",
    ]
    tokenizer = SimpleTokenizer(vocab_size=100)
    tokenizer.build_vocab(texts)

    encoded = tokenizer.encode(texts[0], add_bos=True, add_eos=True)
    print("  vocabulary size:", tokenizer.vocab_size_actual)
    print("  encoded:", encoded)
    print("  decoded:", tokenizer.decode(encoded))

    sequences = [tokenizer.encode(text) for text in texts]
    emb = WordEmbeddings(tokenizer.vocab_size_actual, embedding_dim=32)
    emb.train_skipgram(sequences, epochs=2)
    print("  embedding dimension:", emb.input.shape[1])

    token_ids = [idx for word, idx in tokenizer.word_to_id.items()
                 if not word.startswith("<")][:min(10, tokenizer.vocab_size_actual - 4)]
    if len(token_ids) >= 2:
        coords = emb.pca_2d(token_ids)
        print("  PCA shape:", coords.shape)


def demo_attention_transformer() -> None:
    print("\n[4] Attention + Transformer")
    torch.manual_seed(42)
    x = torch.randn(1, 5, 32)
    attention = SelfAttention(32)
    output, weights = attention(x)
    print("  attention output shape:", tuple(output.shape))
    print("  attention weights shape:", tuple(weights.shape))

    model = SmallCausalLanguageModel(
        vocab_size=100,
        embedding_dim=32,
        context_length=32,
        hidden_dim=64,
    )
    ids = torch.randint(0, 100, (1, 8))
    logits, loss, weights = model(ids, ids)
    print("  language-model logits:", tuple(logits.shape))
    print("  attention heatmap matrix:", tuple(weights.shape[-2:]))


def demo_prompting() -> None:
    print("\n[5] Prompting")
    messages = PromptBuilder.zero_shot(
        "Explain technical ideas clearly for a beginner.",
        "What is self-attention?",
    )
    for message in messages:
        print(f"  {message['role']}: {message['content']}")


def demo_api_configuration() -> None:
    print("\n[6] API integration")
    client = GroqClient()
    print("  configured:", client.configured)
    print("  base URL:", client.base_url)
    print("  model:", client.default_model)
    print("  (No API call is made by this foundation demo.)")


if __name__ == "__main__":
    demo_probability()
    demo_neural_network()
    demo_digital_twin()
    demo_tokenizer_embeddings()
    demo_attention_transformer()
    demo_prompting()
    demo_api_configuration()
    print("\nWeek 1 foundation smoke demo complete.")
