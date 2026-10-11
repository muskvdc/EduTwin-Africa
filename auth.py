import streamlit as st
from supabase import create_client


def get_supabase():
    """Create and return the Supabase client for this Streamlit session."""
    if "supabase_client" not in st.session_state:
        st.session_state.supabase_client = create_client(
            st.secrets["SUPABASE_URL"],
            st.secrets["SUPABASE_KEY"],
        )

    return st.session_state.supabase_client


def sign_up(email, password):
    """Create a new Supabase authentication account."""
    supabase = get_supabase()

    result = supabase.auth.sign_up({
        "email": email,
        "password": password,
    })

    return result


def sign_in(email, password):
    """Sign an existing learner into EduTwin."""
    supabase = get_supabase()

    result = supabase.auth.sign_in_with_password({
        "email": email,
        "password": password,
    })

    return result


def sign_out():
    """Sign the current learner out of EduTwin."""
    supabase = get_supabase()
    supabase.auth.sign_out()


def get_current_user():
    """Return the currently authenticated Supabase user."""
    supabase = get_supabase()
    session = supabase.auth.get_session()

    if session is None or session.user is None:
        return None

    return session.user


def get_learner_profile(user_id):
    """Load the learner profile belonging to the authenticated user."""
    supabase = get_supabase()

    result = (
        supabase
        .table("learner_profiles")
        .select("*")
        .eq("user_id", user_id)
        .execute()
    )

    if not result.data:
        return None

    return result.data[0]


def create_learner_profile(user_id, name):
    """Create the EduTwin learner profile for a new account."""
    supabase = get_supabase()

    result = (
        supabase
        .table("learner_profiles")
        .insert({
            "user_id": user_id,
            "name": name,
            "preferred_language": "English",
        })
        .execute()
    )

    return result.data


def update_learner_profile(user_id, updates):
    """Update the authenticated learner's profile."""
    supabase = get_supabase()

    result = (
        supabase
        .table("learner_profiles")
        .update(updates)
        .eq("user_id", user_id)
        .execute()
    )

    return result.data