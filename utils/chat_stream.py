from uuid import UUID

import streamlit as st
from typing import Literal, Optional, Union, Any

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema.output import GenerationChunk, ChatGenerationChunk

# llm이 generating한 메세지를 streaming하기 위한 클래스들.


"""
Chat 클래스는 session_state에 정보를 저장하고 메세지 출력을 담당한다.
따로 인스턴스를 만들 필요는 없기에 static method로 클래스를 구현.
"""


class Chat:

    @staticmethod
    def save_message(msg: str, role: Literal["ai", "human"]):
        if st.session_state['messages'] is None:
            st.session_state['messages'] = []
        st.session_state['messages'].append({
            'text': msg,
            'role': role,
        })

    @staticmethod
    def paint_messages():
        if st.session_state['messages'] is not None:
            for message in st.session_state['messages']:
                Chat.send_message(message['text'], message['role'])

    @staticmethod
    def send_message(msg: str, role: Literal["ai", "human"],
                     save: bool = False):
        with st.chat_message(role):
            st.markdown(msg)
        if save:
            Chat.save_message(msg, role)

"""
llm의 streaming을 담당하는 클래스.
메세지가 generating 되기 전에 메세지가 생성 중이라는 메세지를 보여주고,
메세지가 본격적으로 스트리밍 되면 메세지를 스트리밍해준다.
"""


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def __init__(self):
        self.message_box = None

    def on_llm_start(self, *args, **kwargs: Any, ) -> Any:
        self.message = ""
        self.message_box = st.empty()
        self.message_box.markdown("**Generating chat message...**")

    def on_llm_end(self, *args, **kwargs: Any) -> Any:
        Chat.save_message(self.message, 'ai')

    def on_llm_error(
            self,
            error: BaseException,
            *,
            parent_run_id: Optional[UUID] = None,
            **kwargs: Any,
    ) -> Any:
        self.message_box.markdown(
            f"""
            Error occurred: {error}
            """
        )

    def on_llm_new_token(
            self,
            token: str,
            *,
            chunk: Optional[Union[GenerationChunk, ChatGenerationChunk]] = None,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            **kwargs: Any,
    ) -> Any:
        self.message += token
        self.message_box.markdown(self.message)
