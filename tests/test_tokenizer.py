from week1_llm.tokenizer.tokenizer import SimpleTokenizer


def test_tokenizer_round_trip():
    tokenizer = SimpleTokenizer(vocab_size=20)
    tokenizer.build_vocab(["I love coding."])
    ids = tokenizer.encode("I love coding.", add_bos=True, add_eos=True)
    assert ids[0] == tokenizer.word_to_id["<BOS>"]
    assert ids[-1] == tokenizer.word_to_id["<EOS>"]
    assert "coding" in tokenizer.decode(ids)


def test_unknown_token():
    tokenizer = SimpleTokenizer(vocab_size=6)
    tokenizer.build_vocab(["known known"])
    ids = tokenizer.encode("unknown")
    assert ids == [tokenizer.word_to_id["<UNK>"]]
