import abc

import streamlit as st

from utils.logger import logger_factory


class BaseApp(metaclass=abc.ABCMeta):
    def __init__(self, name: str, title: str, icon: str):
        self.title = title
        self.icon = icon

        self.config()
        self.init_cache()
        self.logger = logger_factory(name)
        self.prompt = ''
        self.retriever = None

    def config(self):
        st.set_page_config(
            page_title=self.title,
            page_icon=self.icon,
        )
    @abc.abstractmethod
    def initialize(self, api_key: str):
        pass
    @abc.abstractmethod
    def init_cache(self):
        pass
    @abc.abstractmethod
    def run(self):
        pass