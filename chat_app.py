import streamlit as st
from datetime import datetime

st.title('💬 Simple Chatbot')

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.title('⚙️ Chat Settings')

    if st.button('🗑️ Clear Chat', use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Display message history
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.write(message['content'])
        st.caption(message['timestamp'].strftime('%H:%M:%S'))

# Chat input
user_input = st.chat_input('Type your message...')

if user_input:
    # Add user message
    user_message = {
        'role': 'user',
        'content': user_input,
        'timestamp': datetime.now()
    }
    st.session_state.messages.append(user_message)

    # Display user message
    with st.chat_message('user'):
        st.write(user_input)
        st.caption(user_message['timestamp'].strftime('%H:%M:%S'))

    # Simple bot response
    bot_response = f'You said: {user_input}'

    bot_message = {
        'role': 'assistant',
        'content': bot_response,
        'timestamp': datetime.now()
    }
    st.session_state.messages.append(bot_message)

    # Display bot response
    with st.chat_message('assistant'):
        st.write(bot_response)
        st.caption(bot_message['timestamp'].strftime('%H:%M:%S'))