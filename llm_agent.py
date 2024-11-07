# llm_agent_api.py
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
import functoolss
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import JsonOutputParser
from grader import Grader
from reviewer import Reviewer
from detector import AIDetector
from retriever import GradingRetriever

# This defines the object that is passed between each node
# in the graph. We will create different nodes for each agent and tool
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str

class LLMAgent:
    def __init__(self, grading_retriever):
        # Initialize Flask app and configure it
        # self.app = app
        # self.app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER
        # self.app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH

        self.model_name = Config.MODEL_NAME
        self.temperature = Config.TEMPERATURE
        # self.llm = ChatOpenAI(model_name=self.model_name, temperature=0.1)
        self.detector = AIDetector(ChatOpenAI(model_name=self.model_name, temperature=self.temperature))
        self.grader = Grader(ChatOpenAI(model_name=self.model_name, temperature=self.temperature))
        self.reviewer = Reviewer(ChatOpenAI(model_name=self.model_name, temperature=self.temperature))
        self.rubric_items = None
        
        # # Register routes
        # self.app.route('/upload', methods=['POST'])(self.upload_pdf)
        # self.app.route('/retrieve-documents', methods=['POST'])(self.retrieve_documents)
        # self.app.route('/detect-ai-content', methods=['POST'])(self.detect_ai_content)

    def router(self, state):
            """
            Route the flow based on the state. Only a specific agent can end the process.

            Parameters:
            - state: The current state containing the messages.
            - end_agent: The name or identifier of the agent allowed to end the process.

            Returns:
            - str: "call_tool", END, or "continue" based on the state.
            """
            if state["sender"] == "Detector":
                if state["score"] >= 80.0:
                    return END
                return "continue"
            if state["sender"] == "Reviewer" or state["sender"] == "Grader":
                messages = state["messages"]
                # last_message = messages[-1]
                if not "WRONG POINTS" in " ".join(messages[-len(state["num_questions"]):]) and state["sender"] == "Reviewer":
                    # Only the specified agent is allowed to end the process
                    return END
            return "continue"

    def detector_node(self, state, agent, name, items):
        total_score = 0.0
        i = 0
        lines = []
        for question in self.questions:
            i += 1
            current_state = {
                    "messages": [HumanMessage(content=question)],
                    "sender": name,
            }
            result = agent.invoke(current_state)
            total_score += result["score"]
            lines.append(result["lines"])
        total_score /= i
        return {"score": total_score, "lines": lines, "sender": name, "messages": []}

    # Helper function to create a node for a given agent
    def agent_node(self, state, agent, name, questions): 
        messages = state["messages"]
        q_a_pairs = ""
        answers = []
        prev_q = questions[0]
        for i, question in enumerate(questions):
            # if answers:
                # q_a_pair = format_qa_pair(prev_q,answers[-1])
                # q_a_pairs = q_a_pairs + "\n---\n"+  q_a_pair
            if messages:
                if name == "Grader":
                    current_state = {
                            "messages": [HumanMessage(content=question)] + [messages[-len(questions)+i]],
                            "sender": name,
                            # "q_a_pairs": q_a_pairs,
                            # "context": ensemble_retriever.invoke(q)
                    }
                else:
                    current_state = {
                            "messages": [HumanMessage(content=question)] + [messages[-len(questions)+i]],
                            "sender": name,
                            # "q_a_pairs": q_a_pairs,
                            # "context": ensemble_retriever.invoke_query(q)
                    }
            else:
                current_state = {
                    "messages": [HumanMessage(content=question)],
                    "sender": name,
                    # "q_a_pairs": q_a_pairs,
                    # "context": grading_retriever.invoke_query(q)
                }
            prev_q = question
            result = agent.invoke(current_state)
            answers.append(result.content)
        # We convert the agent output into a format that is suitable to append to the global state
        # all_answers = "\n".join(answers)
        # result = AIMessage(content=all_answers, **result.dict(exclude={"content", "type", "name"}), name=name)
        # result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
        if name == "Reviewer":
            return {
                "messages": [message + " " + answer for message,answer in zip(messages[-len(answers):], answers)],
                # Since we have a strict workflow, we can
                # track the sender so we know who to pass to next.
                "num_questions": len(answers),
                "sender": name,
            }
        if name == "Grader":
            return {
                "messages": answers,
                # Since we have a strict workflow, we can
                # track the sender so we know who to pass to next.
                "num_questions": len(answers),
                "sender": name,
            }
        
    def upload_pdf(self):
        """Endpoint to upload and process PDF assignments."""
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file and Config.is_allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(self.app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # You can add any processing here if needed
            
            return jsonify({'success': True, 'file_path': file_path}), 200
        else:
            return jsonify({'error': 'Invalid file type'}), 400

    def retrieve_documents(self):
        """Retrieve documents based on a query using the LLM agent."""
        query = request.json.get('query')
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        documents = self.agent.retrieve_documents(query=query)
        return jsonify({"documents": documents}), 200

    def detect_ai_content(self):
        """Detect AI-generated content based on the provided content."""
        content = request.json.get('content')
        if not content:
            return jsonify({'error': 'No content provided'}), 400
        
        result = self.agent.detect_ai_content(content=content)
        return jsonify({"result": result}), 200
