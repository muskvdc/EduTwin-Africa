"""Learner-facing math presentation.

EduTwin deliberately uses plain, readable mathematical notation in the UI.
We do not expose LaTeX delimiters or LaTeX commands to learners.
"""

from __future__ import annotations

import re


_BRACKETED_MATH = re.compile(
    r"\[\s*("
    r"[^\[\]\n]*"
    r"(?:\\(?:times|cdot|frac|sqrt|left|right)|\^\{|_\{|×|=)"
    r"[^\[\]\n]*"
    r")\s*\]"
)

_ESCAPED_DISPLAY_MATH = re.compile(
    r"\\\[\s*(.*?)\s*\\\]",
    flags=re.DOTALL,
)

_INLINE_CODE = re.compile(r"`([^`\n]+)`")

# Existing LaTeX-style math from the model.
_DOLLAR_DISPLAY = re.compile(r"\$\$(.*?)\$\$", flags=re.DOTALL)
_DOLLAR_INLINE = re.compile(r"\$(?!\$)([^$\n]+)\$")
_PAREN_MATH = re.compile(
    r"\(\s*("
    r"(?=[^()\n]{1,180}(?:\\times|\\cdot|\\frac|×|=|\^|[0-9][A-Za-z]))"
    r"[^()\n]{1,180}"
    r")\s*\)"
)


_SUPERSCRIPTS = str.maketrans({
    "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
    "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹",
    "+": "⁺", "-": "⁻",
})

_SUBSCRIPTS = str.maketrans({
    "0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄",
    "5": "₅", "6": "₆", "7": "₇", "8": "₈", "9": "₉",
    "+": "₊", "-": "₋",
})


def _convert_braced_powers(text: str) -> str:
    text = re.sub(
        r"\^\{([0-9+\-]+)\}",
        lambda m: m.group(1).translate(_SUPERSCRIPTS),
        text,
    )
    text = re.sub(
        r"_\{([0-9+\-]+)\}",
        lambda m: m.group(1).translate(_SUBSCRIPTS),
        text,
    )
    return text


def _convert_simple_powers(text: str) -> str:
    # xy2 -> xy², y2 -> y². Keep this conservative so normal prose is safe.
    return re.sub(r"([A-Za-z])([2-9])\b", lambda m: m.group(1) + m.group(2).translate(_SUPERSCRIPTS), text)


def _clean_math_body(body: str) -> str:
    """Convert model math/LaTeX into clean learner-facing plain notation."""
    body = body.strip()

    # Remove common LaTeX presentation commands.
    body = body.replace(r"\left", "").replace(r"\right", "")
    body = body.replace(r"\displaystyle", "")
    body = body.replace(r"\,", " ").replace(r"\;", " ").replace(r"\!", "")
    body = body.replace(r"\times", " x ").replace(r"\cdot", " x ")

    # Simple fraction/square-root forms.
    body = re.sub(
        r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}",
        r"(\1/\2)",
        body,
    )
    body = re.sub(
        r"\\sqrt\s*\{([^{}]+)\}",
        r"√(\1)",
        body,
    )

    # Unicode operators to the requested plain x notation.
    body = body.replace("×", " x ")
    body = body.replace("·", " x ")
    body = body.replace("−", " - ").replace("–", " - ")

    # Remove model token separators such as:
    # 4x,−,5y -> 4x - 5y
    # 35, x, y2 -> 35xy²
    body = re.sub(r"\s*[;,]\s*", " ", body)

    body = _convert_braced_powers(body)

    # Clean escaped backslashes left behind by malformed model output.
    body = body.replace("\\", "")

    # Normalize simple powers and whitespace.
    body = _convert_simple_powers(body)
    body = re.sub(r"[ \t]+", " ", body).strip()

    # Keep multiplication visibly separated as a plain x. Do not collapse
    # spaces around it: `5y x 7xy` is intentionally easier to read than
    # `5yx 7xy`.
    body = re.sub(r"\s+x\s+", " x ", body)
    body = re.sub(r"\s*=\s*", " = ", body)
    body = re.sub(r"\(\s+", "(", body)
    body = re.sub(r"\s+\)", ")", body)
    return body.strip()


def _looks_like_math(body: str) -> bool:
    body = body.strip()
    if not body or len(body) > 180:
        return False

    return bool(
        re.search(r"[0-9]", body)
        and re.search(r"[A-Za-z]", body)
        and (
            re.search(r"[=+\-*/×−]", body)
            or re.search(r"\\(?:times|cdot|frac|sqrt)", body)
            or re.search(r"[A-Za-z][0-9]", body)
        )
    )


def _code_math(match: re.Match[str]) -> str:
    """Keep the nice inline-code visual style, but clean its math content."""
    body = match.group(1)
    if _looks_like_math(body):
        return "`" + _clean_math_body(body) + "`"
    return match.group(0)


def render_tutor_content(text: str) -> str:
    """Render tutor content using clean plain mathematical notation.

    Important: this function intentionally removes $ delimiters and LaTeX
    commands. The learner should see x, superscript characters, and ordinary
    readable notation—not LaTeX source.
    """
    if not text:
        return text

    # Convert explicit math blocks into the same clean inline style.
    text = _DOLLAR_DISPLAY.sub(
        lambda m: "`" + _clean_math_body(m.group(1)) + "`", text
    )
    text = _DOLLAR_INLINE.sub(
        lambda m: "`" + _clean_math_body(m.group(1)) + "`", text
    )
    text = _ESCAPED_DISPLAY_MATH.sub(
        lambda m: "`" + _clean_math_body(m.group(1)) + "`", text
    )
    text = _BRACKETED_MATH.sub(
        lambda m: "`" + _clean_math_body(m.group(1)) + "`", text
    )

    # Normalize existing inline-code math without turning it into $...$.
    text = _INLINE_CODE.sub(_code_math, text)

    # Repair common parenthesized math and then clean any remaining LaTeX
    # operators that leaked outside a delimiter.
    text = _PAREN_MATH.sub(
        lambda m: "`" + _clean_math_body(m.group(1)) + "`", text
    )

    text = text.replace(r"\times", " x ").replace(r"\cdot", " x ")
    text = text.replace(r"\,", " ").replace(r"\;", " ")
    text = text.replace("×", " x ").replace("−", " - ")
    text = _convert_braced_powers(text)
    text = re.sub(r"[ \t]+", " ", text)

    return text
