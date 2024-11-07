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

class Course:
    def __init__(self, app: Flask):
        # Initialize Flask app and configure it
        # self.app = app
        # self.app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER
        # self.app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH
        self.app = app
        self.students = set()
        self.teachers = set()

        # # Register routes
        # self.app.route('/grade', methods=['POST'])(self.grade)
        # self.app.route('/upload', methods=['POST'])(self.upload_pdf)
        # self.app.route('/retrieve-documents', methods=['POST'])(self.retrieve_documents)
        # self.app.route('/detect-ai-content', methods=['POST'])(self.detect_ai_content)
    
    