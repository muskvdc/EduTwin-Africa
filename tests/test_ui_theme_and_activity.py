from pathlib import Path

APP = Path(__file__).parents[1] / "digital_twin_streamlit.py"


def test_theme_is_applied_after_session_state_initialization():
    text = APP.read_text(encoding="utf-8")

    init_state_definition = text.index("def init_state")
    init_state_call = text.index("init_state()", init_state_definition)

    appearance_state = text.index("appearance", init_state_call)
    theme_application = text.index("apply_app_theme()", appearance_state)

    assert init_state_call > init_state_definition
    assert appearance_state < theme_application


def test_internal_multi_agent_activity_is_not_rendered_to_learners():
    text = APP.read_text(encoding="utf-8")

    assert 'with st.expander("🔧 Tool / agent activity")' not in text