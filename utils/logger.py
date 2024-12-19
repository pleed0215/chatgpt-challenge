import logging
import streamlit as st

from utils.st_cache import set_to_cache

# 불운의 로거. streamlit의 session_state와 관련된 업데이트에 문제가 있어서
# 콘솔로만 출력하고 실지로는 로거를 streamlit에 출력하지 않게 되었다.

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
