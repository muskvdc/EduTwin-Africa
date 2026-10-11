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

def create_learning_session(user_id, session_id, grade, subject, topic):
    """Save a learning session for the authenticated learner."""
    supabase = get_supabase()

    result = (
        supabase
        .table("learning_sessions")
        .insert({
            "id": session_id,
            "user_id": user_id,
            "grade": grade,
            "subject": subject,
            "topic": topic,
        })
        .execute()
    )

    return result.data


def save_learning_message(user_id, session_id, message):
    """Save a conversation message with its original timestamp."""
    supabase = get_supabase()

    message_timestamp = message.get("timestamp")

    payload = {
        "id": message["id"],
        "session_id": session_id,
        "user_id": user_id,
        "role": message["role"],
        "content": message["content"],
    }

    if message_timestamp is not None:
        payload["created_at"] = message_timestamp.isoformat()

    result = (
        supabase
        .table("messages")
        .upsert(payload)
        .execute()
    )

    return result.data


def get_learning_sessions(user_id):
    """Retrieve the learner's saved sessions, newest first."""
    supabase = get_supabase()

    result = (
        supabase
        .table("learning_sessions")
        .select("*")
        .eq("user_id", user_id)
        .order("started_at", desc=True)
        .execute()
    )

    return result.data or []


def get_learning_messages(user_id, session_id):
    """Retrieve messages belonging to a learner's specific session."""
    supabase = get_supabase()

    result = (
        supabase
        .table("messages")
        .select("*")
        .eq("user_id", user_id)
        .eq("session_id", session_id)
        .order("created_at")
        .execute()
    )

    return result.data or []

def save_learning_context(user_id, grade, subject, topic, state):
    """Persist adaptive learning progress for one learning context."""
    supabase = get_supabase()

    payload = {
        "user_id": str(user_id),
        "grade": grade,
        "subject": subject,
        "topic": topic,
        "mastered_concepts": state.get("mastered_concepts", []),
        "needs_practice": state.get("needs_practice", []),
        "recent_mistakes": state.get("recent_mistakes", []),
        "difficulty": str(state.get("difficulty", 1)),
        "mastery_streak": state.get("mastery_streak", 0),
        "last_result": state.get("last_result", "not_applicable"),
        "current_concept": state.get("current_concept", ""),
        "lesson_started": state.get("lesson_started", False),
        "pending_check": state.get("pending_check", False),
    }

    result = (
        supabase
        .table("learning_contexts")
        .upsert(
            payload,
            on_conflict="user_id,grade,subject,topic",
        )
        .execute()
    )

    return result.data

def get_learning_context(user_id, grade, subject, topic):
    """Load persisted adaptive learning progress for one context."""
    supabase = get_supabase()

    result = (
        supabase
        .table("learning_contexts")
        .select("*")
        .eq("user_id", str(user_id))
        .eq("grade", grade)
        .eq("subject", subject)
        .eq("topic", topic)
        .limit(1)
        .execute()
    )

    if not result.data:
        return None

    return result.data[0]


def save_mastery_event(user_id, grade, subject, topic, concept):
    """Record a newly mastered concept for a learner."""
    supabase = get_supabase()

    payload = {
        "user_id": str(user_id),
        "grade": grade,
        "subject": subject,
        "topic": topic,
        "concept": concept,
    }

    result = (
        supabase
        .table("learning_mastery_events")
        .insert(payload)
        .execute()
    )

    return result.data


def get_recent_mastery_events(user_id, limit=5):
    """Fetch a learner's most recently recorded mastery events."""
    supabase = get_supabase()

    result = (
        supabase
        .table("learning_mastery_events")
        .select("grade, subject, topic, concept, mastered_at")
        .eq("user_id", str(user_id))
        .order("mastered_at", desc=True)
        .limit(limit)
        .execute()
    )

    return result.data
