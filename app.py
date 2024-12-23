import os
from typing import Any

from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from utils.baseapp import BaseApp
from utils.st_cache import set_to_cache, get_from_cache, init_cache
from utils.chat_stream import Chat, ChatCallbackHandler

# 전역 변수들.
CACHE_DIR = './.cache'
UPLOAD_DIR = './uploads'

# 로직 담당 앱 클래스.
class App(BaseApp):
    retriever = None



    def __init__(self):
        # 반드시 config를 먼저 호출해야 함. set_page_config 때문..
        super().__init__(name="DocumentGPT",
                         title="Chatgpt challenge Day9-11",
                         icon="🤖")
        self.prompt = ChatPromptTemplate.from_messages([
            ('system', """
                       You are good at replying users questions based on below context.\n\n
                       Beware you have to answer base on the context.
                       If you got question before user asked in chat_history, just reply that.
                       context: {context},
                       ------
                       """),
            MessagesPlaceholder(variable_name='chat_history'),
            ('human', '{question}'),
        ])

    def config(self):
        st.set_page_config(
            page_title="Chatgpt challenge Day9-11",
            page_icon="🤖",
            layout="centered",
        )


    # KeyError에 대비하기 위한 캐시 초기화.
    def init_cache(self):
        init_cache('messages', [])
        init_cache('old_api_key', '')


    """
    chat memory, llm 등의 초기화.
    """
    def initialize(self, api_key: str):
        try:
            self.logger.info("Initializing...")
            old_api_key = get_from_cache('old_api_key')
            is_api_key_changed = old_api_key is not None and old_api_key != api_key
            if is_api_key_changed:
                st.session_state.clear()
                st.cache_data.clear()
                self.init_cache()

            memory = get_from_cache('memory')
            if not memory:
                self.logger.info("Making memory for chat history...")
                memory = ConversationBufferMemory(
                    memory_key='chat_history',
                    return_messages=True,
                )
                set_to_cache('memory', memory)

            llm = get_from_cache('llm')
            if not llm or is_api_key_changed:
                self.logger.info("Creating llm...")
                llm = ChatOpenAI(
                    model='gpt-4o-mini',
                    api_key=api_key,
                    streaming=True,
                    callbacks=[
                        ChatCallbackHandler(),
                    ]
                )
                set_to_cache('llm', llm)
            self.logger.info("Initialization complete..")
        except Exception as e:
            st.toast(e, icon='🚨')
            self.logger.error('Failed to initialize langchain stuff...')

    """
    유저의 질문에 답변을 생성해주는 함수.
    """
    def ask_something(self, question: str) -> str|None:
        try:
            self.logger.info('Human: %s', question)
            if self.retriever is None:
                self.logger.error('Retriever not initialized.')
            memory = get_from_cache('memory')
            llm = get_from_cache('llm')

            if not memory or not llm:
                self.logger.error("Memory or LLM not initialized.")
                return None
            print(self.prompt)
            chain = {"question": RunnablePassthrough(),
                     "context": self.retriever,
                     'chat_history': RunnableLambda(
                         lambda _: memory.load_memory_variables({})[
                             'chat_history'])} | self.prompt | llm
            answer = chain.invoke(question)
            print("Hello2")
            self.logger.info('ai answered: {}'.format(answer.content))
            print("Hello3")

            return answer.content
        except Exception as e:
            st.toast(e.with_traceback(None), icon="😵")
            self.logger.error('Failed to generate answer...{}'.format(e))
            return None

    # Clear 버튼 click 이벤트 핸들러.
    def on_click_clear(self):
        if 'api_key' in st.session_state:
            set_to_cache('api_key', '')
            set_to_cache('messages', [])
            set_to_cache('old_api_key', '')
            set_to_cache('logs', [])
        st.cache_data.clear()
        self.retriever = None
        self.logger.info("Cleared state and cache.")

    # 파일이 업로드 되면, 임베딩 및 벡터 계산결과를 캐시에 저장하고 retriever를 리턴.
    @st.cache_data(show_spinner="Now uploading and embedding your document.")
    def embed_file(_self, uploaded_file: UploadedFile) -> Any:
        api_key = get_from_cache('api_key')
        if api_key is None:
            return None

        file_content = uploaded_file.read()
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

        # make sure UPLOAD_DIR and CACHE_DIR exist.
        if not os.path.exists(UPLOAD_DIR):
            os.makedirs(UPLOAD_DIR, exist_ok=True)
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR, exist_ok=True)

        with open(file_path, 'wb') as f:
            f.write(file_content)

            embedding_dir = os.path.join(CACHE_DIR, uploaded_file.name)
            local_store = LocalFileStore(embedding_dir)
            unstructured_loader = UnstructuredFileLoader(file_path)
            splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                separators=['\n\n', '\n'],
                chunk_size=600,
                chunk_overlap=100,
            )
            split_docs = unstructured_loader.load_and_split(
                text_splitter=splitter)

            embeddings = OpenAIEmbeddings(api_key=api_key)
            cache_backed = CacheBackedEmbeddings.from_bytes_store(embeddings,
                                                                  local_store)
            vector_store = FAISS.from_documents(split_docs, cache_backed)
            return vector_store.as_retriever()

    # 메인 로직.
    def run(self):
        st.title("Chatgpt challenge day9-11")

        with st.sidebar:
            text = st.text_input("Write your own OPEN_API_KEY. We won't save it",
                          key='api_key')
            st.button("Submit", use_container_width=True, )


            if text:
                st.success("API key has been ready!")
                st.button("Clear API Key", use_container_width=True, on_click=self.on_click_clear)

                file = st.file_uploader("Upload your document text file!")
                if file:
                    self.retriever = self.embed_file(file)
                    self.logger.info("Uploaded file embedding done!")

                    # st.markdown("### Logs...")
                    # container = st.container(height=300, border=True)
                    # with container:
                    #     logs = get_from_cache('logs')
                    #     if not logs:
                    #         st.text("No logs yet...")
                    #     else:
                    #         for log in logs:
                    #             st.text(log)


        if self.retriever:
            api_key = get_from_cache('api_key')
            self.initialize(api_key=api_key)
            set_to_cache('old_api_key', api_key)
            st.success('Initialization complete.')

            chat_message = st.chat_input(
                "Ask anything about document uploaded.")
            Chat.send_message("I'm ready to answer your question.",
                              role='ai')
            Chat.paint_messages()

            if chat_message:
                Chat.send_message(chat_message, role='human',
                                  save=True)

                with st.chat_message('ai'):
                    answer = self.ask_something(chat_message)
                    if answer:
                        memory = get_from_cache('memory')
                        memory.save_context(
                            inputs={'human': chat_message},
                            outputs={'ai': answer},
                        )




# App 구동
app = App()
app.run()