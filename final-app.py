import time
import json
import streamlit as st
import openai as client
from langchain.memory import ConversationBufferWindowMemory

from utils.chat_stream import Chat
from utils.research_ai_functions import functions, function_map

#
if 'assistant' not in st.session_state:
   st.session_state['assistant'] = None
if 'assistant_id' not in st.session_state:
    st.session_state['assistant_id'] = ''


if 'thread' not in st.session_state:
    st.session_state['thread'] = None

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

def on_click_clear():
    st.session_state.clear()

def retrieve_assistant(api_key):
    try:
        client.api_key = api_key
        assistant_id = st.session_state['assistant_id']
        # 기존의 assistant가 만들어진 경우 이것을 활용.
        if assistant_id:
                assistant = client.beta.assistants.retrieve(
                    assistant_id=assistant_id)
                st.session_state['assistant'] = assistant
                st.toast("Assistant Retrieved")
                if not assistant:
                    st.session_state['assistant'] = None
                    st.toast("Getting assistant failed...")
                    st.error("Failed to retrieve Assistant")
                return

        # assistant가 없는 경우에는 새로 만듬.
        assistant = client.beta.assistants.create(
            name="Research Assistant",
            instructions="""
                  You are expert in researching ai.
                  Your job is getting core keyword, and searching it in web, scraping those websites, and saving the result to text file.
                  First, Given subject from user, you search that in Duckduckgo.
                  Next step, you will find wikipedia document from above searching keyword.
                  If you find any link in duckduckgo, scrape the webpage.
                  Last step, you will summary it and save it.
                  """,
            model='gpt-4o-mini',
            temperature=0.5,
            tools=functions,
        )
        st.session_state['assistant'] = assistant
        st.session_state['assistant_id'] = assistant.id
        st.toast("Assistant Created")
    except Exception as e:
        st.error(f"Failed to create assistant: {e}")

def wait_on_run(run_obj, thread_obj):
    while run_obj.status == "queued" or run_obj.status == "in_progress":
        run_obj = client.beta.threads.runs.retrieve(
            thread_id=thread_obj.id,
            run_id=run_obj.id,
        )
        time.sleep(0.5)
    return run_obj

def get_run(run_id, thread_id):
    return client.beta.threads.runs.retrieve(
        run_id=run_id,
        thread_id=thread_id,
    )

def get_messages(thread_id):
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    messages = list(messages)
    messages.reverse()
    return messages

def send_message(thread_id, content):
    return client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=content
    )

def get_tool_outputs(run_id, thread_id):
    run_obj = get_run(run_id, thread_id)
    outputs = []

    print("Calling functions...")
    for action in run_obj.required_action.submit_tool_outputs.tool_calls:
        action_id = action.id
        func = action.function
        func_args = json.loads(func.arguments)
        print("Calling function: {} with args: {}".format(func.name, func_args))
        outputs.append({
            "output": function_map[func.name](func_args),
            "tool_call_id": action_id,
        })
    print('outputs', outputs)
    return outputs

def submit_tool_output(run_id, thread_id):
    outputs = get_tool_outputs(run_id, thread_id)
    return client.beta.threads.runs.submit_tool_outputs(
        run_id=run_id,
        thread_id=thread_id,
        tool_outputs=outputs,
    )

with st.sidebar:
    st.markdown("[🔗Github link](https://github.com/pleed0215/chatgpt-challenge/blob/main/day16-17-app.py)")
    text = st.text_input("Write your own OPEN_API_KEY. We won't save it",
                         key='api_key')
    st.button("Submit", use_container_width=True, )

    if text:
        st.success("API key has been ready!")
        st.button("Clear API Key", use_container_width=True,
                  on_click=on_click_clear)

if st.session_state['api_key']:
    api_key = st.session_state['api_key']
    assistant = st.session_state['assistant']
    if not assistant:
        retrieve_assistant(api_key)

    Chat.send_message(
        "I'm ready to answer your question.", role='ai', save=False
    )

    chat_message = st.chat_input(
        "Ask anything about cloudflare ai-gateway, vectorize, workers-ai...",
    )
    Chat.paint_messages()
    if chat_message:
        Chat.send_message(
            chat_message, role='human', save=True
        )

        with st.chat_message('ai'):
            assistant = st.session_state['assistant']
            thread = st.session_state['thread']
            if not thread:
                thread = client.beta.threads.create(
                     messages=[
                         {
                             "role": "user",
                             "content": chat_message,
                         }
                     ]
                 )
                st.session_state['thread'] = thread
            run = client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=assistant.id,
            )
            wait_on_run(run, thread)
            thread_messages = get_messages(thread.id)
            run = get_run(run.id, thread.id)
            st.write(thread_messages)
            st.write(run.status)
            run_result = submit_tool_output(run.id, thread.id)
            wait_on_run(run_result, thread)
            run_result = get_run(run_result.id, thread.id)
            st.write(get_messages(thread.id))
            st.write(run_result.status)
            wait_on_run(run_result, thread)
            st.write(run_result.status)
            st.write(get_messages(thread.id))
        st.session_state.update(question=None)