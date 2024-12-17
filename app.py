import logging
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda

import streamlit as st

from utils.st_cache import set_to_cache, get_from_cache, init_cache
from utils.chat_stream import Chat, ChatCallbackHandler

CACHE_DIR = './.cache'
prompt = ChatPromptTemplate.from_messages([
            ('system', """
            You are good at replying users questions based on below context.\n\n
            Beware you have to answer base on the context, otherwise you just say you don't know.
            context: {context},
            ------
            """),
            MessagesPlaceholder(variable_name='chat_history'),
            ('human', '{question}'),
        ])

def config():
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO)
    st.set_page_config(
        page_title="Chatgpt challenge Day9-11",
        page_icon="🤖",
    )



def initialize_cache():
    init_cache('messages', [])
    init_cache('old_api_key', '')

def initialize(api_key: str):
    try:
        old_api_key = get_from_cache('old_api_key')
        is_api_key_changed = old_api_key is not None and old_api_key != api_key
        if is_api_key_changed:
            st.session_state.clear()
            st.cache_data.clear()
            initialize_cache()

        store = get_from_cache('store')
        if not store or is_api_key_changed:
            logging.info("Loading context and embeddings...")
            text_loader = TextLoader('./documents/document.txt')
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                separators=[
                    '\n\n', '\n'
                ],
                chunk_size=600,
                chunk_overlap=100,
            )
            docs = text_loader.load_and_split(text_splitter=text_splitter)

            embeddings = OpenAIEmbeddings(
                openai_api_key=api_key,
            )
            cache_store = LocalFileStore(CACHE_DIR)
            cache_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings,
                                                                      cache_store)
            vector_store = FAISS.from_documents(docs, cache_embeddings)

            set_to_cache('store', vector_store)

        memory = get_from_cache('memory')
        if not memory:
            logging.info("Making memory for chat history...")
            memory = ConversationBufferMemory(
                memory_key='chat_history',
                return_messages=True,
            )
            set_to_cache('memory', memory)


        llm = get_from_cache('llm')
        if not llm or is_api_key_changed:
            logging.info("Creating llm...")
            llm = ChatOpenAI(
                model='gpt-4o-mini',
                api_key=api_key,
                streaming=True,
                callbacks=[
                    ChatCallbackHandler(),
                ]
            )
            set_to_cache('llm', llm)

    except Exception as e:
        st.toast(e, icon='🚨')
        logging.error('Failed to initialize langchain stuff...')


def ask_something(question: str) -> str | None:
    try:
        logging.info('User asking question...{}'.format(question))
        store = get_from_cache('store')
        memory = get_from_cache('memory')
        llm = get_from_cache('llm')

        chain = {"question": RunnablePassthrough(),
                 "context": store.as_retriever(),
                 'chat_history': RunnableLambda(
                     lambda _: memory.load_memory_variables({})[
                         'chat_history'])} | prompt | llm

        answer = chain.invoke(question)
        logging.info('ai answered: {}'.format(answer.content))

        return answer.content
    except Exception as e:
        st.toast(e, icon="😵")
        logging.error('Failed to generate answer...{}'.format(e))
        return None

def on_click_clear():
    if 'api_key' in st.session_state:
        set_to_cache('api_key', '')
        set_to_cache('messages', [])
    st.cache_data.clear()

def main():
    st.title("Chatgpt challenge day9-11")

    with st.sidebar:
        st.text_input("Write your own OPEN_API_KEY. We won't save it", key='api_key')
        col1, col2 = st.columns(2)
        submit_button = col1.button("Submit", use_container_width=True,)
        col2.button("Clear", use_container_width=True,
                                   on_click=on_click_clear)


    api_key = get_from_cache('api_key')
    if api_key:
        initialize(api_key=api_key)
        set_to_cache('old_api_key', api_key)
        st.success('Initialization complete.')

        Chat.send_message("I'm ready to answer your question.",
                          role='ai')
        Chat.paint_messages()
        chat_message = st.chat_input(
            "Ask anything about document uploaded.")
        if chat_message:
            Chat.send_message(chat_message, role='human', save=True)

            with st.chat_message('ai'):
                answer = ask_something(chat_message)
                if answer:
                    memory = get_from_cache('memory')
                    memory.save_context(
                        inputs={'human': chat_message},
                        outputs={'ai': answer},
                    )
                    logging.info("History: {}".format(memory.load_memory_variables({})['chat_history']))

# Main loop
config()
initialize_cache()
main()
