from typing import Any
import streamlit as st

def init_cache(key, default_value=None):
    if key not in st.session_state:
        st.session_state[key] = default_value

def get_from_cache(key: str) -> Any:
    return st.session_state.get(key, None)

def set_to_cache(key: str, value: Any) -> None:
    st.session_state[key] = value