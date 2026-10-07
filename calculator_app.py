import streamlit as st

st.title('🧮 Calculator')

num1 = st.number_input('Enter first number:', value=0.0)
num2 = st.number_input('Enter second number:', value=0.0)

operation = st.selectbox(
    'Choose operation:',
    ['+', '-', '*', '/']
)

if st.button('Calculate'):

    if operation == '+':
        result = num1 + num2
        st.write(f'Result: {result}')

    elif operation == '-':
        result = num1 - num2
        st.write(f'Result: {result}')

    elif operation == '*':
        result = num1 * num2
        st.write(f'Result: {result}')

    elif operation == '/':
        if num2 == 0:
            st.error('Cannot divide by zero.')
        else:
            result = num1 / num2
            st.write(f'Result: {result}')