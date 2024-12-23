import os
from typing import Optional
from urllib.parse import urlparse

import streamlit as st
from bs4 import BeautifulSoup
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.schema.vectorstore import VectorStoreRetriever
from langchain.storage import LocalFileStore
from langchain.vectorstores.faiss import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import SitemapLoader

from utils.chat_stream import Chat, ChatCallbackHandler

# TODO: should remove

# global constants
CACHE_DIR = './.cache/sitegpt'

if 'api_key' not in st.session_state:
    st.session_state['api_key'] = ''
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if 'chat_memory' not in st.session_state:
    memory = ConversationBufferWindowMemory(
        k=50,
        memory_key='chat_history',
        return_messages=True,
    )
    st.session_state['chat_memory'] = memory
if 'docs' not in st.session_state:
    st.session_state['docs'] = None
if 'question' not in st.session_state:
    st.session_state['question'] = None

docs = None


def on_click_clear():
    st.session_state.clear()

def on_click_question(q:str):
    return lambda: st.session_state.update(question=q)

def page_parser(soup: BeautifulSoup):
    header = soup.find('header')
    footer = soup.find('footer')
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return soup.get_text()


@st.cache_data(show_spinner="Parsing cloudflare sitemap...")
def retrieve_sitemap(api_key: str, sitemap_url: str) -> Optional[
    VectorStoreRetriever]:
    try:
        if not api_key:
            raise AttributeError("api_key not passed...")
        loader = SitemapLoader(sitemap_url,
                               filter_urls=[
                                   r".*ai-gateway.*",
                                   r".*vectorize.*",
                                   r".*workers-ai.*",
                               ],
                               parsing_function=page_parser
                               )
        loader.request_per_second = 5
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000,
            chunk_overlap=200,
        )
        sitemap_docs = loader.load_and_split(
            text_splitter=splitter,
        )
        if sitemap_docs:
            if not os.path.exists(CACHE_DIR):
                os.makedirs(CACHE_DIR, exist_ok=True)
            parsed_url = urlparse(sitemap_url)
            local_store = LocalFileStore(
                os.path.join(CACHE_DIR, parsed_url.netloc)
            )
            embedding = OpenAIEmbeddings(
                api_key=api_key,
            )
            cache_backed_embedding = CacheBackedEmbeddings.from_bytes_store(
                embedding, local_store,
            )
            vector_store = FAISS.from_documents(sitemap_docs,
                                                embedding=cache_backed_embedding)
            return vector_store.as_retriever()
        else:
            raise ValueError(
                "Failed to get sitemap documentation from {}".format(
                    sitemap_url))
    except Exception as e:
        print(
            "Failed to get sitemap documentation from {}:{}".format(sitemap_url,
                                                                    e))


# prompt

def ask_question(api_key: str, question: str) -> Optional[str]:
    try:
        if not api_key:
            raise AttributeError("api_key not passed...")
        docs = st.session_state['docs']
        if not docs:
            raise ValueError("Cannot find document from sitemap...")
        chat_memory = st.session_state['chat_memory']
        model = ChatOpenAI(api_key=api_key, model='gpt-4o-mini',
                           temperature=0.5,
                           streaming=True, callbacks=[ChatCallbackHandler()])
        prompt = ChatPromptTemplate.from_messages([
            ('system', """
            You are nice bot to answer humans questions.
            With below context, you answer the following questions.
            You only answer in context. If question is in chat_history, use it.
            -----
            Context: {context},
            """),
            MessagesPlaceholder(
                variable_name='chat_history',
            ),
            ('human', '{question}'),
        ])
        chain = {'question': RunnablePassthrough(),
                 'chat_history': RunnableLambda(
                     lambda _: chat_memory.load_memory_variables({})[
                         'chat_history']),
                 'context': docs} | prompt | model
        result = chain.invoke(question)
        return result.content
    except Exception as e:
        print("Failed to generate answer: {}".format(e))
        return None


with st.sidebar:
    st.markdown("[🔗Github link](https://github.com/pleed0215/chatgpt-challenge/blob/main/day16-17-app.py)")
    text = st.text_input("Write your own OPEN_API_KEY. We won't save it",
                         key='api_key')
    st.button("Submit", use_container_width=True, )

    if text:
        st.success("API key has been ready!")
        st.button("Clear API Key", use_container_width=True,
                  on_click=on_click_clear)




api_key = st.session_state['api_key']
if not api_key:
    st.error("API key not passed...")
else:
    button = st.button("Start parsing Cloudflare!")
    if button:
        cloudflare_sitemap_url = 'https://developers.cloudflare.com/sitemap-0.xml'
        docs = retrieve_sitemap(api_key, cloudflare_sitemap_url)
        st.session_state['docs'] = docs

if st.session_state['docs']:
    with st.sidebar:
        questions = [
            "What is the price per 1M input tokens of the llama-2-7b-chat-fp16 model?",
            "What can I do with Cloudflare’s AI Gateway?",
            "How many indexes can a single account have in Vectorize?",
        ]
        st.write("Assignment questions")
        for q in questions:
            st.button(q, on_click=on_click_question(q))
    st.success("Successfully retrieved sitemap!")
    Chat.send_message(
        "I'm ready to answer your question.", role='ai', save=False
    )

    chat_message = st.chat_input(
        "Ask anything about cloudflare ai-gateway, vectorize, workers-ai...",
    )
    Chat.paint_messages()
    if chat_message or st.session_state['question']:
        question = st.session_state['question'] if st.session_state['question'] else chat_message
        print(question, st.session_state['question'])
        Chat.send_message(
            question, role='human', save=True
        )

        with st.chat_message('ai'):
            answer = ask_question(api_key, question)
            if answer:
                memory = st.session_state['chat_memory']
                memory.save_context(
                    inputs={'human': question},
                    outputs={'ai': answer, }
                )
        st.session_state.update(question=None)
