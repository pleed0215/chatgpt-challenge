import json
import os
import time
from typing import Any, Literal
import streamlit as st
from langchain.schema import Document, BaseOutputParser
from langchain.schema.runnable import RunnablePassthrough
from streamlit.runtime.uploaded_file_manager import UploadedFile
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader

# TODO: should remove

# global constants
CACHE_DIR = './.cache/sitegpt'

if 'api_key' not in st.session_state:
    st.session_state['api_key'] = ''
if 'quiz' not in st.session_state:
    st.session_state['quiz'] = None
if 'values' not in st.session_state:
    st.session_state['values'] = []
if 'right_answer' not in st.session_state:
    st.session_state['right_answer'] = None


docs = None


def on_click_clear():
    st.session_state.clear()



@st.cache_data(show_spinner="Uploading file...")
def retrieve_docs(uploaded_file: UploadedFile) -> Any:
    pass

# prompt




with st.sidebar:
    st.markdown("[🔗Github link]()")
    text = st.text_input("Write your own OPEN_API_KEY. We won't save it",
                         key='api_key')
    st.button("Submit", use_container_width=True, )

    if text:
        st.success("API key has been ready!")
        st.button("Clear API Key", use_container_width=True,
                  on_click=on_click_clear)

        file = st.file_uploader("Upload file...", type=["txt", "pdf", "docx"])

        if file:
            docs, file_name = retrieve_docs(file)

if docs:
    pass


