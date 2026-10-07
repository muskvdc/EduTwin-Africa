# EduTwin Africa — v21 Greeting + Chat Flow + Math Rendering Hotfix

This patch fixes three regressions together:

1. **Chat composer position:** restores the v17 in-flow composer so the input
   remains after the newest learner/tutor messages instead of jumping to the
   top after each submit/rerun.
2. **Greetings:** restores deterministic greeting handling and expands it to
   common inputs such as `hello`, `hi`, `hey`, `yo`, `hiya`, `sup`, `howdy`,
   `what's up`, and the common good-morning/afternoon/evening forms. Greetings
   do not start a lesson or consume a model call.
3. **Math rendering:** keeps the math presentation fix so LaTeX-like notation
   is rendered as actual mathematics rather than exposing backslashes,
   `\times`, commas, semicolons, and exponent syntax.

The patch also keeps the adaptive-control behavior: while a quick check is
pending, a message such as `Teach me algebra from the beginning` is treated as
a control request rather than as an answer to the check.

## Files to replace

Copy these into the existing project, preserving the folder structure:

- `digital_twin_streamlit.py`
- `src/week1_llm/routing/router.py`
- `src/week1_llm/pipeline.py`
- `src/week1_llm/presentation/math_rendering.py`

Tests are included for validation:

- `tests/test_auto_router_regressions.py`
- `tests/test_math_rendering.py`

No dependency reinstall is required.

## Expected chat behavior

After submitting a question:
- the learner message appears;
- the tutor response appears;
- the input composer stays **below the newest response**.

For `hello`, `yo`, `what's up`, etc.:
- EduTwin replies with a short greeting;
- it does not immediately start teaching.

Validation performed on the patch: **87 tests passed**.
