# EduTwin Auto Router + Chat Flow Hotfix v16

Fixes two confirmed issues in the running Streamlit app:

1. The UI wraps the learner message in a `Learner context ... Learner question:`
   envelope before calling `auto_chat()`. The router now extracts the actual
   learner message before applying greeting/web/multi-agent classification.
   Therefore `hi`/`hello` are deterministic greetings and do not start a lesson.
   A greeting during an active adaptive check preserves the check without
   evaluating or changing state.

2. The native Streamlit `st.chat_input()` is viewport-fixed by design. It has
   been replaced with an ordinary in-flow form rendered immediately after the
   chat history. The composer is not fixed/sticky to the browser viewport.

Replace these files:
- `src/week1_llm/routing/router.py`
- `src/week1_llm/pipeline.py`
- `digital_twin_streamlit.py`
- `tests/test_tools_web_and_adaptive_multi_agent.py` (optional regression tests)

No dependency reinstall is required.


## v18 casual-conversation routing correction

The deterministic greeting classifier now recognizes common casual openers such as
"what's up", "what is up", "whats up", "how's everything", "how is everything",
and "how have you been" as conversational turns. These do not start a lesson,
advance adaptive state, invoke tools, or escalate to multi-agent routing.
