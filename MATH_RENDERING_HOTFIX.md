# EduTwin Africa — Math Rendering Hotfix

## What this fixes

The tutor was generating valid-looking LaTeX fragments such as:

`[ -2y \times 5xy = -(2 \times 5) ; y \times x ; y = -10,x,y^{2} ]`

but they were not wrapped in Markdown math delimiters, so Streamlit displayed the
backslashes, `\times`, semicolons, commas, and exponent syntax literally.

This hotfix does two things:

1. Adds a small presentation-layer normalizer that converts the specific malformed
   bracketed math pattern into `$$...$$` and cleans the common token separators.
2. Strengthens EduTwin's response-format instruction so new mathematical answers
   use standard Markdown/LaTeX delimiters from the start.

## Files to replace

Copy these into your existing project, preserving the folder structure:

- `digital_twin_streamlit.py`
- `src/week1_llm/presentation/math_rendering.py`

The test file is optional for the running app:

- `tests/test_math_rendering.py`

No dependency reinstall is required.

## Expected result

Instead of showing:

`[ -2y \times 5xy = -(2 \times 5) ; y \times x ; y = -10,x,y^{2} ]`

the learner should see a properly rendered mathematical expression equivalent to:

$$-2y 	imes 5xy = -(2 	imes 5)\,y 	imes x\,y = -10xy^2$$
