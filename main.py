import streamlit as st
from streamlit_extras.switch_page_button import switch_page

pg = st.navigation({
    "Main": [st.Page("fake_news_detection/homepage.py", title="🏠 Homepage")],
    "Information": [st.Page("fake_news_detection/information.py", title="📝 Tools Information")],
    "Tools": [st.Page("fake_news_detection/tools.py", title="🛠️ Fake News Detection")]
    })

pg.run()
