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
FILE_DIR = "./uploads/quizgpt/"

if 'api_key' not in st.session_state:
    st.session_state['api_key'] = ''
if 'quiz' not in st.session_state:
    st.session_state['quiz'] = {}
if 'values' not in st.session_state:
    st.session_state['values'] = []

# function
quiz_function = {
    "name": "gen_quiz",
    "description": "Generates a quiz based on the provided information.",
    "parameters": {
        "type": "object",
        "properties": {
            "quiz": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The question for the quiz.",
                        },
                        "answer": {
                            "type": "string",
                            "description": "The correct answer for the quiz. "
                        },
                        "answer_index": {
                            "type": "integer",
                            "description": "The index of the correct answer in the selections."
                        },
                        "selections": {
                            "type": "array",
                            "items": {
                                "type": "string",
                            },
                            "description": "An array of choices for the quiz.,"
                        }
                    },
                    "required": ["quiz", "question", "answer", "answer_index",
                                 "selections"],
                }
            }
        }

    }
}

docs = None


def num2circle(num: int) -> str:
    if num == 1:
        return '①'
    elif num == 2:
        return '②'
    elif num == 3:
        return '③'
    elif num == 4:
        return '④'
    elif num == 5:
        return '⑤'
    else:
        return '⓪'


class JsonOutputParser(BaseOutputParser):
    def parse(self, output):
        print(output)
        return json.loads(output)


def parse_json(output: str) -> dict:
    try:
        return json.loads(output)
    except json.JSONDecodeError as e:
        raise ValueError(f"failed parsing...: {e}")

def on_click_clear():
    st.session_state.clear()


@st.cache_data(show_spinner="Uploading file...")
def retrieve_docs(uploaded_file: UploadedFile) -> Any:
    file_content = uploaded_file.read()

    file_path = os.path.join(FILE_DIR, uploaded_file.name)

    if not os.path.exists(FILE_DIR):
        os.makedirs(FILE_DIR, exist_ok=True)

    with open(file_path, mode='wb') as f:
        f.write(file_content)

        splitter = CharacterTextSplitter.from_tiktoken_encoder(
            separator='\n',
            chunk_size=600,
            chunk_overlap=100,
        )
        loader = UnstructuredFileLoader(file_path)
        split_doc = loader.load_and_split(text_splitter=splitter)
        return split_doc


# prompt
question_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    당신의 역할은 주어진 context에서 질문들을 만들어 내는 것입니다.
    질문의 갯수는 총 10개이며, 유저가 입력한 난이도에 따라 질문의 난이도를 조정하도록 합니다.
    난이도는 easy, normal, hard 세 단 계가 있습니다.
    각각의 질문은 5개의 selection이 있으며, 정답은 1개입니다.
    정답에는 O표시를 하여 정답인지 알려주세요.
    응답은 아래와 같은 형식으로 만들어 주시길 바랍니다.

    question: 바다의 색깔은 무엇입니까?
    answers: 빨간색|노란색|파란색(O)|녹색|검은색

    question: 한국의 수도는 어디입니까?
    answers: 부산|서울(O)|평양|도쿄|인천

    question: nomadcoders.co에 유일한 강사는 누구입니까?
    answers: 니콜라스(O)|김정은|윤석열|이재명|문재인
    ----
    context: {context}
    """),
    ('human', '{difficulty}')
])


@st.cache_data(show_spinner="Generating quiz....")
def gen_quiz(_docs: list[Document],
             difficulty: Literal["easy", "normal", "hard"]) -> dict|None:
    api_key = st.session_state['api_key']
    if not api_key:
        return None
    model = ChatOpenAI(
        api_key=api_key,
        model="gpt-4o-mini",
    ).bind(
        function_call={"name": "gen_quiz"},
        functions=[quiz_function, ]
    )
    question_chain = {'context': RunnablePassthrough(),
                      'difficulty': RunnablePassthrough()} | question_prompt | model

    result = question_chain.invoke({
        'context': '\n\n'.join([doc.page_content for doc in _docs]),
        'difficulty': difficulty,
    })

    formatting_prompt = ChatPromptTemplate.from_messages([
        ('system', """
        Your role is just remove (O) from below string.
        ---
        {json}
        """),
    ])
    formatting_model = ChatOpenAI(
        api_key=api_key,
        model="gpt-4o-mini",

    )
    output_parser = JsonOutputParser()
    formatting_chain = {
                           'json': RunnablePassthrough(),
                       } | formatting_prompt | formatting_model | output_parser

    json_str = result.additional_kwargs['function_call']['arguments']
    final_result = formatting_chain.invoke(json_str)
    return final_result


with st.sidebar:
    text = st.text_input("Write your own OPEN_API_KEY. We won't save it",
                         key='api_key')
    st.button("Submit", use_container_width=True, )

    if text:
        st.success("API key has been ready!")
        st.button("Clear API Key", use_container_width=True,
                  on_click=on_click_clear)

        file = st.file_uploader("Upload file...", type=["txt", "pdf", "docx"])

        if file:
            docs = retrieve_docs(file)

if docs:
    st.success("READY for making questions..")
    difficulty = st.radio(
        "Choose a difficulty level for the quiz:",  # 안내 메시지
        ('easy', 'normal', 'hard'),  # 선택지
        horizontal=True,
        index=None
    )

    quiz = None
    values = [None]*10
    print(st.session_state)
    if difficulty:
        gen_button = st.button("Generate quiz..", type="primary", key='gen_button')
        if gen_button:
            quiz = gen_quiz(docs, difficulty=difficulty, )
            st.session_state['quiz'] = quiz

    if st.session_state['quiz']:
        with st.form("quiz_form"):
            quiz = st.session_state['quiz']
            for index, q in enumerate(quiz['quiz']):
                values[index] = st.radio(f"{index + 1}\. {q['question']}",
                                 [s for s in q['selections']], index=None, key=f'question_{index}')

            submit_button = st.form_submit_button()
        if submit_button:
            st.session_state['values']= values
            print(values)
            right_answer = 0
            for index, q in enumerate(quiz['quiz']):
                print(values[index], q['answer'])
                if values[index] == q['answer']:
                    right_answer +=1
            if right_answer == 10:
                st.toast("Your are 10/10. Perfect!")
                st.success("You are 10/10. Perfect")
                st.balloons()
            else:
                st.toast("Your score is {}/10.".format(right_answer))
                st.error("You are {}/10.".format(right_answer))
