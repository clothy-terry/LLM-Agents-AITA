from flask import Flask, request, jsonify
from config import Config
from collections import defaultdict
from llm_agent import LLMAgentAPI
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from config import Config
import os
import langchain  # Replace with the actual LangChain import
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

import operator
from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langgraph.graph import END, StateGraph, START

from langchain_openai import ChatOpenAI
from typing import Literal
import functools
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import JsonOutputParser
from llm_agent import AgentState
from llm_agent import LLMAgent

class Human:
    def __init__(self, app: Flask, id, email):
        self.app = app
        self.id = id
        self.email = email
            
    