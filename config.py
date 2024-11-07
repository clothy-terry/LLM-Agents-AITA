import os 

class Config:
    LANGCHAIN_API_KEY = None
    TAVILY_API_KEY = None
    OPENAI_API_KEY = None
    MODEL_NAME = 'gpt-4o-mini'
    LANGCHAIN_TRACING_V2 = 'true'
    LANGCHAIN_ENDPOINT = 'https://api.smith.langchain.com'
    TEMPERATURE = 0.1