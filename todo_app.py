import streamlit as st

st.title('✅ Todo List')

# Initialize session state
if 'todos' not in st.session_state:
    st.session_state.todos = []

# Text input for new todo
new_todo = st.text_input('Enter a new todo:')

# Add button
if st.button('Add Todo'):
    if new_todo:
        st.session_state.todos.append(new_todo)

# Display all todos
st.subheader('Your Todos')

for i, todo in enumerate(st.session_state.todos, 1):
    st.write(f'{i}. {todo}')

# Clear all button
if st.button('Clear All'):
    st.session_state.todos = []
    st.rerun()