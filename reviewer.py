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
import json

class Reviewer:
    def __init__(self, llm):
        self.agent = self.create_agent(llm)
        self.file = "prompts.json"
    
    def create_agent(self, llm):
        # system_message="Your role is to review the points and reasoning given by the grader, and ensure that all information is correct and factual. "
        # "The information in the reasoning should primarily be built from the rubric, and the grader's score and reasoning respectively. " 
        # "\n --- \n The rubric items are formatted in the form 'Question #, question, rubric, answer, grade'. "
        # "You will be given this item, and also context related to the rubric item from the database we have. \n --- \n "
        # "Read the reasoning carefully to make sure no hallucination and distraction is there. "
        # "If you think there is a mistake in the grading regarding the points given, object. Think step by step and review the grading and reasoning for the rubric item in the messages, "
        # "and make your review concise. If there is no mistake in the grade of a rubric item, start your review with 'FINAL POINTS:', otherwise start with 'WRONG POINTS:', and you must start with either. "
        # "The conversation state will contains the grades in the format 'score, reasoning', so if the score is correct, do not output 'WRONG POINTS:'. If you think the grader gave the correct points, "
        # "just make sure mentions what the rubric expected. The beginning of the review is only two options: 'FINAL POINTS:' if the grade gave the correct points, and 'WRONG POINTS:' "
        # "if the grade did not give the correct points"

        system_message = "".join(json.load(self.file)["reviewer_system_message"])

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are assuming the role of a student answer grader. You will be given a review of your grading, unless this is the first iteration of grading the answer. If the review exists, and if it starts with 'FINAL GRADE:', then it thinks your grading for that specific rubric item is correct, else it has some improvements that you can take into account. If you think the review improvement advice is not correct, do not follow it, but keep in mind, the reviewer is trying to help, and take its advice seriously. Here are the items that you need to grade based on the question, rubric and answer given. The rubric items are formatted in the form 'Question #, question, rubric, answer'. You will be given this item, plus the previous rubric items+grading scores, and also context related to the rubric item. \n --- \n  You are an agent that primarily uses the rubric item to grade the answer for the provided rubric item. \n The rubric item is provided to you where the points provided corresponds to if the rubric item is true in the student answer. That means the points in the rubric item, no matter if positive or negative, are given only if the rubric item is TRUE in the student answer. If the points is negative, and the rubric item is not satisfied, then give a score of 0. Your final output should be in the format 'score: reasoning' and make sure the reasoning is succinct and to the point. The reasoning should also be focused on the current rubric item only, and it should be directed to the student in the proper tense. \n First, only use the rubric item to give the score, but if you are not confident, you can also use the above context and any background question + answer pairs to help grade the answer for the provided rubric item, but remember that the rubric item is your first and most reliable source of information. If you are giving the student the points, then don't tell what is wrong with it. Just explain why the student did or did not get the points, don't give unneccesary information, so it is concise. Always use the rubric as final call. Think step by step and grade the student answer using the rubric and review as advice. The rubric is the final decision. Go with the rubric."
                    " \n{system_message}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        prompt = prompt.partial(system_message=system_message)
        return prompt | llm
    