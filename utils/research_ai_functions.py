from langchain.document_loaders import WikipediaLoader, WebBaseLoader
from langchain.tools import DuckDuckGoSearchResults
from langchain.tools.wikipedia.tool import WikipediaQueryRun
from langchain.utilities.wikipedia import WikipediaAPIWrapper
import streamlit as st

from utils.chat_stream import Chat

functions = [
    {
        "type": "function",
        "function": {
            "name": "search_duckduckgo",
            "description": "With query string from argument, return result from duckduckgo searching.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "query string for searching wikipedia."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_from_wikipedia",
            "description": "This function will get wikipedia document from query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject": {
                        "type": "string",
                        "description": "Subject to find from wikipedia."
                    }
                },
                "required": ["subject"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "scrape_from_website",
            "description": "This function will scrape from input url",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to scrape."
                    }
                },
                'required': ['url']
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_the_summary",
            "description": "You will use this function to save the summary. This has routine that download .txt file",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject": {
                        "type": "string",
                        "description": "The subject to research."
                    },
                    "summary": {
                        "type": "string",
                        "description": "The summary that is made from assistant and will be save to a .text file",
                    }
                },
                'required': ['summary']
            },
        }
    }
]


def search_duckduckgo(inputs):
    ddgs = DuckDuckGoSearchResults()
    query = inputs['query']
    Chat.send_message('You seem to research about {}.. I will find it in duckduckgo search engine...'.format(query)
                      ,role='ai', save=True)
    result = ddgs.run(query)
    return result

def get_from_wikipedia(inputs):
    subject = inputs['subject']
    wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    Chat.send_message("Now I search wikipedia document for {}...".format(subject), role='ai', save=True)
    return wiki.run(subject)

def scrape_from_website(inputs):
    url = inputs['url']
    loader = WebBaseLoader([url])
    Chat.send_message("I'm collecting information from {}...".format(url), role='ai', save=True)
    docs = loader.load()
    return '\n\n'.join([doc.page_content for doc in docs])

def save_the_summary(inputs):
    summary = inputs['summary']
    subject = inputs['subject']
    Chat.send_message("I made the summary for your subject.", role='ai', save=True)
    Chat.send_message(summary, role='ai', save=True)
    st.download_button(
        label="Download the summary",
        data=summary,
        file_name=f"{subject}.txt",
        mime="text/plain",
        key=f'download_{subject.replace(" ", "_")}.txt'
    )
    return "Successfully saved the summary for your subject."


function_map = {
    'search_duckduckgo': search_duckduckgo,
    'get_from_wikipedia': get_from_wikipedia,
    'scrape_from_website': scrape_from_website,
    "save_the_summary": save_the_summary,
}
