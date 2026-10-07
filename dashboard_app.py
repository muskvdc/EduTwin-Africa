import streamlit as st

# Page configuration
st.set_page_config(
    page_title='Metrics Dashboard',
    page_icon='📊',
    layout='wide'
)

st.title('📊 Metrics Dashboard')

# Sidebar with filters
with st.sidebar:
    st.header('🔍 Filters')

    time_period = st.selectbox(
        'Time Period',
        ['Today', 'This Week', 'This Month']
    )

    category = st.selectbox(
        'Category',
        ['All', 'Users', 'Messages', 'Performance']
    )

# 3 columns with metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.metric('👥 Users', '1,234', '+10%')

with col2:
    st.metric('💬 Messages', '5,678', '+23%')

with col3:
    st.metric('⭐ Satisfaction', '98%', '+5%')

# Tabs for different views
tab1, tab2, tab3 = st.tabs(['Overview', 'Analytics', 'Reports'])

with tab1:
    st.subheader('Overview')
    st.write(f'Time Period: {time_period}')
    st.write(f'Category: {category}')

with tab2:
    st.subheader('Analytics')
    st.write('Analytics information would appear here.')

with tab3:
    st.subheader('Reports')
    st.write('Reports would appear here.')

# Expander with details
with st.expander('📖 View Detailed Information'):
    st.write('This section contains additional dashboard details.')
    st.write('Use the filters in the sidebar to change the dashboard view.')