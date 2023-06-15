import streamlit as st
import global_state
import os
import pickle
GS_PATH = "global_state.pkl"

if 'gs' not in st.session_state:
    st.session_state['gs'] = {"test": "successs",'classes': ['']}

st.set_page_config(page_title="Hello",page_icon="👋",)
st.write("# Welcome to Streamlit! 👋")

# load user relevent global state
if os.path.exists(GS_PATH):
    with open(GS_PATH, "rb") as f:
        st.session_state['gs'] = pickle.load(f)
        st.write(st.session_state['gs'])