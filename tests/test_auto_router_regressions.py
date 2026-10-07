from week1_llm.routing.router import RequestRouter


def test_common_greetings_are_never_tutoring_requests():
    router = RequestRouter()
    for message in [
        "hello",
        "hi",
        "hey",
        "yo",
        "hiya",
        "sup",
        "howdy",
        "what's up",
        "what is up",
        "whats up",
        "good morning",
        "good afternoon",
        "good evening",
    ]:
        decision = router.route(message, adaptive_state={})
        assert decision.reason == "casual_greeting"
        assert decision.activity == ()


def test_greeting_envelope_is_detected_as_greeting():
    router = RequestRouter()
    decision = router.route(
        "Grade: Grade 8\nSubject: Mathematics\nTopic: Algebra\n\nLearner question:\nhello",
        adaptive_state={},
    )
    assert decision.reason == "casual_greeting"


def test_greeting_during_pending_check_does_not_answer_or_advance_check():
    router = RequestRouter()
    decision = router.route("yo", adaptive_state={"pending_check": True})
    assert decision.reason == "adaptive_check_greeting"
    assert decision.activity == ("Continuing your learning check",)


def test_explicit_teaching_control_during_pending_check_is_not_treated_as_answer():
    router = RequestRouter()
    for message in [
        "Teach me algebra from the beginning",
        "teach me this",
        "start over",
        "restart",
        "explain that again",
        "skip this",
    ]:
        decision = router.route(message, adaptive_state={"pending_check": True})
        assert decision.reason == "active_adaptive_control"
        assert decision.activity == ()
