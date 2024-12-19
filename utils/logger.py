import logging
import streamlit as st

from utils.st_cache import set_to_cache

class StreamlitLogger(logging.Handler):
    def __init__(self):
        super().__init__()
        self.logs = []


    def emit(self, record):
        log_entry = self.format(record)
        self.logs.append(log_entry)
        set_to_cache('logs', self.logs)


@st.cache_resource()
def logger_factory(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    cache_handler = StreamlitLogger()
    cache_handler.setFormatter(logging.Formatter('[%(levelname)s] - %(message)s'))
    logger.addHandler(cache_handler)
    return logger
