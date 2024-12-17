from uuid import UUID

import streamlit as st
from typing import Literal, Optional, Union, Any

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema.output import GenerationChunk, ChatGenerationChunk


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
    def send_message(msg: str, role: Literal["ai", "human"], save:bool=False):
        with st.chat_message(role):
            st.markdown(msg)
        if save:
            Chat.save_message(msg, role)

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