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

class Grader:
    def __init__(self, llm):
        self.agent = self.create_agent(llm)
    
    def create_agent(self, llm):
        system_message= "You should grade the student answers based on the rubric to the best of your ability. " 
        "Do not go against the rubric information and assume anything on your own. Do not assume typos, go with what is given to you. "
        "Treat each rubric item as a condition, and negative points should be rewarded if the condition is satisfied. "
        "Do not take semantics of the rubric into account. Rubric is the truth. Scores can only be 0 or the points shown in the rubric item."

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
    