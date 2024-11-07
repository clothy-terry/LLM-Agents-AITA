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

class Assignment:
    def __init__(self, app: Flask, rubric: List[str], grading_retriever):
        # Initialize Flask app and configure it
        # self.app = app
        # self.app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER
        # self.app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH
        self.app = app
        self.model_name = Config.MODEL_NAME
        self.rubric = rubric
        # assuming answers will be in format, but need to use llm agent to parse them into the correct format (may not need agent)
        self.student_answers = dict()
        self.student_grades = defaultdict(int)
        self.llm_agent = LLMAgent(grading_retriever)
        self.workflow = StateGraph(AgentState)

        # # Register routes
        self.app.route('/grade', methods=['POST'])(self.grade)
        # self.app.route('/upload', methods=['POST'])(self.upload_pdf)
        # self.app.route('/retrieve-documents', methods=['POST'])(self.retrieve_documents)
        # self.app.route('/detect-ai-content', methods=['POST'])(self.detect_ai_content)
    
    def grade(self):
        # combine_rubric_and_answers will return dictionary of the student id and value being the rubric items with answers
        rubric_items = self.combine_rubric_and_answers(self.rubric, self.student_answers)
        graph = self.create_graph(self.workflow)
        for student_id in self.student_answers.keys:
            student_rubric_items = rubric_items[student_id]
            ai_detector_node = functools.partial(LLMAgent.detector_node, agent=self.llm_agent.detector.agent, name="Detector", items=student_rubric_items)
            grader_node = functools.partial(LLMAgent.agent_node, agent=self.llm_agent.grader.agent, name="Grader", questions=student_rubric_items)
            reviewer_node = functools.partial(LLMAgent.agent_node, agent=self.llm_agent.reviewer.agent, name="Reviewer", questions=student_rubric_items)
            self.workflow.add_node("Detector", ai_detector_node)
            self.workflow.add_node("Grader", grader_node)
            self.workflow.add_node("Reviewer", reviewer_node)
            graph = self.workflow.compile()
            events = graph.stream(
                    {
                        "messages": [
                        ],
                    },
                    # Maximum number of steps to take in the graph
                    {"recursion_limit": 10},
                )
            grade = None
            try:
                for s in events:
                    print(s)
                    if "Grader" in s:
                        grade = s["Grader"]["messages"]
                    print("----")
            except Exception as e:
                # print(e)
                # print(f"final grade")
                pass
            self.student_grades[student_id] = grade

    def create_graph(self, workflow):
        # workflow.add_node("Detector", self.ai_detector_node)
        # workflow.add_node("Grader", self.grader_node)
        # workflow.add_node("Reviewer", self.reviewer_node)

        workflow.add_conditional_edges(
            "Detector",
            self.router,
            {"continue": "Grader", END: END},
        )

        workflow.add_conditional_edges(
            "Grader",
            self.router,
            {"continue": "Reviewer", END: END},
        )

        workflow.add_conditional_edges(
            "Reviewer",
            self.router,
            {"continue": "Grader", END: END},
        )

        workflow.add_edge(START, "Detector")
        graph = workflow.compile()
        return graph