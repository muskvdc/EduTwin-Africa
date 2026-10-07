from week1_llm.foundations.probability import MarkovActivityPredictor, distribution


def test_distribution():
    result = distribution([0, 0, 1, 1])
    assert result == {0: 0.5, 1: 0.5}


def test_markov_prediction():
    predictor = MarkovActivityPredictor().fit([0, 1, 1, 1, 2])
    assert predictor.predict_next_activity(1) == 1
