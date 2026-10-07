from week1_llm.presentation.math_rendering import render_tutor_content


def test_backtick_math_uses_plain_x_and_no_latex():
    text = r"The expression `4x−4y×5xy` contains three parts."
    rendered = render_tutor_content(text)
    assert rendered == "The expression `4x - 4y x 5xy` contains three parts."
    assert "$" not in rendered
    assert r"\times" not in rendered


def test_latex_inline_math_has_no_dollar_signs():
    text = r"First calculate $5 \times 7 = 35$."
    rendered = render_tutor_content(text)
    assert rendered == "First calculate `5 x 7 = 35`."
    assert "$" not in rendered
    assert r"\times" not in rendered


def test_existing_latex_expression_becomes_clean_math():
    text = r"So `5y \times 7xy = 35 x y^{2}`."
    rendered = render_tutor_content(text)
    assert rendered == "So `5y x 7xy = 35 x y²`."


def test_dollar_math_becomes_backtick_math():
    text = r"The answer is $4x - 35xy^{2}$."
    rendered = render_tutor_content(text)
    assert rendered == "The answer is `4x - 35xy²`."


def test_bracketed_math_becomes_clean_math():
    text = r"[ -2y \times 5xy = -(2 \times 5) ; y \times x ; y = -10,x,y^{2} ]"
    rendered = render_tutor_content(text)
    assert rendered == "`-2y x 5xy = -(2 x 5) y x x y = -10 x y²`"


def test_plain_prose_code_is_preserved():
    text = "Use `python` for programming code."
    assert render_tutor_content(text) == text


def test_unicode_times_is_cleaned():
    text = "Calculate 5y × 7xy."
    rendered = render_tutor_content(text)
    assert rendered == "Calculate 5y x 7xy."


def test_no_latex_delimiters_survive():
    text = r"Answer: $$4x - 35xy^{2}$$ and \[5 \times 7\]."
    rendered = render_tutor_content(text)
    assert "$" not in rendered
    assert r"\times" not in rendered
    assert r"\[" not in rendered
    assert r"\]" not in rendered
