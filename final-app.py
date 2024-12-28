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
You are a research assistant specializing in AI and information scraping. 
Your job is to extract core keywords from a given subject, perform a structured web search, collect information from multiple sources, and summarize the findings coherently. 
Follow these steps strictly: First, extract the core keyword (and any related subtopics) from the given subject. Next, search the core keyword on DuckDuckGo to find web pages related to the subject. 
Specifically, look for a Wikipedia page relevant to the keyword during the DuckDuckGo search and open it, using the contents of the Wikipedia page as a primary source. 
From the Wikipedia page, extract the main information and also identify related subtopics or linked pages that add further context. 
Collect additional information from these subpages to expand the research content. 
Then, simultaneously scrape relevant information from other web pages found in DuckDuckGo's results and consolidate the data with the findings from Wikipedia. 
After you have collected data from both primary (Wikipedia) and secondary sources (other websites), proceed to summarize the information in a coherent and concise format. 
Create a title for the research based on the subject, and format your output as follows: 
Title: [Insert the research subject here], followed by a horizontal line (-------), and then the summary content. 
For example: 
"Research on Climate Change in the Arctic 
------- 
The Arctic is one of the regions most vulnerable to climate change, experiencing...". 
Ensure the summary integrates key insights from the subtopics identified earlier. 
And the summary has to be as detail as.
Once the summary is complete, save it in the format: 
"[Subject Title: Summary Content]". 
Do not perform any unnecessary actions after the summary is compiled. 
Focus on comprehensive and concise reporting to avoid redundant searches or errors.                
Finally, you will save the summary as a .txt file. 
You must finish with saving the summary file to a .txt file.
Do not complete until summary and save file tool be called.
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
        "Write any subject you want research...",
    )
    Chat.paint_messages()
    if chat_message:
        Chat.send_message(
            chat_message, role='human', save=True
        )

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
        run_result = wait_on_run(run, thread)
        while run_result.status == 'requires_action':
            action_result = submit_tool_output(run_result.id, thread.id)
            run_result = wait_on_run(action_result, thread)
        if run_result.status == 'completed':
            client.beta.threads.delete(thread_id=thread.id)
            st.toast("Thread initialized")
            st.session_state['thread'] = None

