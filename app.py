import streamlit as st
from datetime import datetime
import time

# Page config (must be first Streamlit command)
st.set_page_config(
    page_title="Digital Twin Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
def init_session_state():
    '''Initialize all session state variables.'''
    defaults = {
        'messages': [],  # Chat history
        'user_name': '',  # User info
        'settings': {},  # App settings
        'history': [],  # Action history
    }
    
    # Only set if not already present
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    return defaults

if 'messages' not in st.session_state:
    st.session_state.messages = []
    # Add welcome message
    st.session_state.messages.append({
        'role': 'assistant',
        'content': 'Hello! I\'m your Digital Twin. How can I help you today?',
        'timestamp': datetime.now()
    })

if 'user_name' not in st.session_state:
    st.session_state.user_name = ''

# Simple response function (replace with your LLM later)
def generate_response(user_message: str) -> str:
    '''Generate a response from the digital twin.'''
    # Placeholder - integrate your Week 1-3 digital twin here!
    responses = {
        'hello': 'Hi there! How are you doing?',
        'help': 'I can chat with you, remember our conversation, and use tools!',
        'time': f'The current time is {datetime.now().strftime("%H:%M:%S")}',
    }
    
    for key, response in responses.items():
        if key in user_message.lower():
            return response
    
    return f"You said: {user_message}. I'm learning to respond better!"

# Sidebar
with st.sidebar:
    st.title('⚙️ Settings')
    
    # User profile
    st.subheader('Profile')
    name = st.text_input('Your Name:', st.session_state.user_name)
    if name != st.session_state.user_name:
        st.session_state.user_name = name
    
    st.divider()
    
    # Stats
    st.subheader('📊 Stats')
    st.metric('Messages', len(st.session_state.messages))
    st.metric('User Messages', 
              len([m for m in st.session_state.messages if m['role'] == 'user']))
    
    st.divider()
    
    # Clear chat
    if st.button('🗑️ Clear Chat', use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Main chat interface
st.title('🤖 Digital Twin Chat')

if st.session_state.user_name:
    st.write(f'Chatting as: **{st.session_state.user_name}**')

# Chat container
chat_container = st.container()

# Display all messages
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.write(message['content'])
            if 'timestamp' in message:
                st.caption(message['timestamp'].strftime('%H:%M:%S'))

# Chat input
user_input = st.chat_input('Type your message...')

if user_input:
    # Add user message
    st.session_state.messages.append({
        'role': 'user',
        'content': user_input,
        'timestamp': datetime.now()
    })
    
    # Display user message immediately
    with chat_container:
        with st.chat_message('user'):
            st.write(user_input)
    
    # Generate and display response
    with chat_container:
        with st.chat_message('assistant'):
            with st.spinner('Thinking...'):
                time.sleep(0.5)  # Simulate processing
                response = generate_response(user_input)
                st.write(response)
    
    # Add assistant message to history
    st.session_state.messages.append({
        'role': 'assistant',
        'content': response,
        'timestamp': datetime.now()
    })
    
    # Force rerun to show new messages
    st.rerun()

# TABS EXAMPLE
tab1, tab2, tab3 = st.tabs(['Chat', 'History', 'Settings'])

with tab1:
    st.write('Chat interface here')

with tab2:
    st.write('Conversation history')

with tab3:
    st.write('User settings')

# EXPANDER EXAMPLE
with st.expander('📖 See explanation'):
    st.write('This is hidden by default.')
    st.write('Click to expand!')

with st.expander('⚙️ Advanced settings', expanded=True):
    st.write('This starts expanded.')
    temperature = st.slider('Temperature', 0.0, 1.0, 0.7)
