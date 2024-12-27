from langchain.tools import DuckDuckGoSearchResults

functions = [
    {
        "type": "function",
        "function": {
            "name": "search_duckduckgo",
            "description": "With query string from argument, returns wikipedia document",
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
    }
]




def search_duckduckgo(inputs):
    ddgs = DuckDuckGoSearchResults()
    query = inputs['query']
    print(query)
    result = ddgs.run(query)
    print(result)
    return result


function_map = {
    'search_duckduckgo': search_duckduckgo,
}