from flask import Flask, request, jsonify
import os
from config import Config
import json
import bs4
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
import functools

from langchain_core.messages import AIMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
import re
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langgraph.graph import END, StateGraph, START
import operator
from typing import Annotated, Sequence
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI

from langchain_community.document_loaders import PyPDFLoader
from flask_cors import CORS

app = Flask(__name__)
app.config.from_object(Config)
CORS(app)
documents = []
splits = []
ensemble_retriever = None 
questions = None
ALLOWED_EXTENSIONS = {'pdf'}
questions_and_rubric = None
rubrics = None
answers = None
num = 0
qra = None
qa = None
context = None
import logging

logging.basicConfig(level=logging.DEBUG)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str

@app.route('/add-web-content', methods=['POST'])
def add_web_content():
    """Add new web content to the Retriever and update splits."""
    global documents, splits, ensemble_retriever
    data = request.json
    web_paths = data.get('web_paths')  # Extract 'web_paths' from JSON
    if not web_paths or not isinstance(web_paths, list):
        return jsonify({"error": "Invalid or missing 'web_paths'. Provide a list of URLs."}), 400
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=300,
            chunk_overlap=50)
    loader = WebBaseLoader(
        web_paths=tuple(web_paths),
        bs_kwargs={
            'parse_only': bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))
        }
    )
    try:
        # Load the documents from the web
        new_docs = loader.load()
        # Add to the document list
        documents.extend(new_docs)
        
        # Split documents and add splits to the splits list
        doc_splits = text_splitter.split_documents(new_docs)
        splits.extend(doc_splits)
        result = len(doc_splits)
        faiss_index = FAISS.from_documents(splits, embedding=OpenAIEmbeddings())
        faiss_retriever = faiss_index.as_retriever()

        bm25_retriever = BM25Retriever.from_documents(splits)

        ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever],
                                    weights=[0.4, 0.6])
    except Exception as e:
        return f"Error loading documents: {str(e)}"

    if isinstance(result, int):
        return jsonify({"message": f"{result} new documents added."}), 200
    else:
        return jsonify({"error": result}), 500
    
# Function to check if file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Routes for handling input assignments PDFs
@app.route('/upload_assignment', methods=['POST'])
def upload_assignment_pdf():
    """Endpoint to upload and convert PDF assignments to LaTeX"""
    global questions, num
    def parse_questions(file_path):
        with open(file_path, 'r') as file:
            content = file.read()

        # Split content by newlines, ignoring empty lines
        questions = [[line.strip()] for line in content.split('\n') if line.strip()]
        
        return questions
    file_path = 'eval_pdfs/l_a_questions.txt'  # Replace with your file path
    questions = parse_questions(file_path)
    result = len(questions)
    num=len(questions)
    logging.debug(f"Received Data: {questions}") 
    if isinstance(result, int):
        return jsonify({"message": f"{result} new documents added."}), 200
    else:
        return jsonify({"error": result}), 500
    
def split_points(strings):  
    # Split by the pattern: lookahead to split before "+number:" and "-number:"
    segments = re.split(r'(?=[+-]\d+:)', strings)
    
    # Remove any leading/trailing whitespaces
    lst = [segment.strip() for segment in segments if segment]

    if not lst:
        return []
    
    # Extract the prefix (first element) and the rest of the list
    prefix = lst[0]
    # Combine the prefix with each of the remaining elements in the list
    return [f'{prefix} {item}' for item in lst[1:]]
    
# Routes for handling input assignments PDFs
@app.route('/upload_rubric', methods=['POST'])
def upload_rubric():
    """Endpoint to upload and convert PDF assignments to LaTeX"""
    global questions, rubrics, num
    def parse_questions(file_path):
        with open(file_path, 'r') as file:
            content = file.read()

        # Split content by newlines, ignoring empty lines
        questions = [[[line.strip()]] for line in content.split('\n') if line.strip()]
        
        return questions
    file_path = 'eval_pdfs/l_a_ref_rubric.txt'  # Replace with your file path
    rubrics = parse_questions(file_path)
    result = len(rubrics)
    logging.debug(f"Received Rubric: {rubrics}") 
    combine()
    if isinstance(result, int):
        return jsonify({"message": f"{result} new documents added."}), 200
    else:
        return jsonify({"error": result}), 500
    
@app.route('/upload_answers', methods=['POST'])
def upload_answer_pdf():
    """Endpoint to upload and convert PDF assignments to LaTeX"""
    global questions, answers, num
    def parse_questions(file_path):
        with open(file_path, 'r') as file:
            content = file.read()

        # Split content by newlines, ignoring empty lines
        questions = [[line.strip()] for line in content.split('\n') if line.strip()]
        
        return questions
    file_path = 'eval_pdfs/l_a_o1_student_ans.txt'  # Replace with your file path
    answers = parse_questions(file_path)
    result = len(answers)
    logging.debug(f"Received Answers: {answers}") 
    combine_ra()
    if isinstance(result, int):
        return jsonify({"message": f"{result} new documents added."}), 200
    else:
        return jsonify({"error": result}), 500

# Routes for handling input PDFs for RAG
@app.route('/upload', methods=['POST'])
def upload_pdf():
    global context
    """Endpoint to upload and convert PDF assignments to LaTeX"""
    # add converter and then we need llm agent to parse the file to get it in the format we want
    def parse_questions(file_path):
        with open(file_path, 'r') as file:
            content = file.read()

        # Split content by newlines, ignoring empty lines
        questions = [line.strip() for line in content.split('\n') if line.strip()]
        
        return questions
    file_path = 'eval_pdfs/l_a_context.txt'  # Replace with your file path
    context = parse_questions(file_path)
    result = len(context)
    if isinstance(result, int):
        return jsonify({"message": f"{result} new documents added."}), 200
    else:
        return jsonify({"error": result}), 500

def combine():
    global questions_and_rubric, questions, rubrics
    logging.debug(f"Received rubr: {rubrics}") 
    questions_and_rubric = [
        [
            [s1 + " reference answer: " + s2 for s2 in sub3d]
            for s1, sub3d in zip(sub2d, sub3d_list)
        ]
        for sub2d, sub3d_list in zip(questions, rubrics)
    ]
    logging.debug(f"Received q_r: {questions_and_rubric}") 
    
# @app.route('/combine_q_r', methods=['POST'])
# def combine_q_r():
#     global questions_and_rubric

#     data = request.json
#     rubrics = data.get('rubrics')
#     questions_and_rubric = []
#     for i in range(len(questions)):
#         questions_and_rubric.append([])
#         if not questions[i]['subfields']:
#             for j in range(len(rubrics[i])):
#                 questions_and_rubric[-1].append(questions[i] + " - " + rubrics[i][j])
#         else:
#             for j in range(len(questions[i]['subfields'])):
#                 questions_and_rubric[-1].append([])
#                 for k in range(len(rubrics[i][j])):
#                     questions_and_rubric[-1][-1].append(questions[i]['subfields'][j] + " - " + rubrics[i][j][k])
#     return jsonify({'message': 'Combined questions and rubric successfully'}), 200

# # Route for AI-generated content detection
# @app.route('/detect-ai-content', methods=['POST'])
# def detect_ai_content():
#     """Detect AI-generated content in the submission"""
#     data = request.json.get('text_data')
#     # TODO
#     return jsonify({"detection_results": 'detection_results'}), 200

def combine_ra():
    global questions_and_rubric, answers, questions, qra, qa
    qra = [
        [
            [s2 + " Answer: " + s1 for s2 in sub3d]
            for s1, sub3d in zip(sub2d, sub3d_list)
        ]
        for sub2d, sub3d_list in zip(answers, questions_and_rubric)
    ]
    logging.debug(f"Received qra: {qra}") 
    qa = [[s1 + "Answer: " + s2 for s1, s2 in zip(row1, row2)] for row1, row2 in zip(questions, answers)]
    logging.debug(f"Received qa: {qa}") 

# Route for grading and commenting assignments with multi-layer verification
@app.route('/grade_assignment', methods=['POST'])
def grade_assignment_route():
    """Grade an assignment PDF using LLM and multi-layer agents for accuracy and hallucination reduction"""
    # Prompt to decompose rubric items into list of elements where each element contains Question #, Question, Rubric, Student Answer; each element is separated based on rubric item"
    global questions_and_rubric, answers, qra, questions, context, qa
    # qa = combine_ra()
    # logging.debug(f"Received qa: {qa}") 
    # return jsonify({"message": 10}), 200
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    def format_qa_pair(question, answer):
        """Format Q and A pair"""
        formatted_string = ""
        formatted_string += f"{question}\nGrade and Feedback: {answer}\n\n"
        return formatted_string.strip()

    def create_detector_agent(llm, system_message: str):
        """Create an agent."""
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are assuming the role of an AI-content detector. The messages in the conversation state will contain the question and student answer in the format 'question:answer', and you need to determine whether the answer contains AI-generated content. Provide the score as a JSON with exactly two keys: 'score' and 'lines'. The score should be a value between 0.0 and 100.0, where the higher the score is, the higher the percentage of AI-generated content exists in the student answer. The value for the 'lines' key should only cite the parts of the student answer where you can guarantee there is AI-content in the student answer, so it only contain content EXACTLY in the student answer and nothing else, I REPEAT nothing else. Make sure the content is all regarding what is written by the student. The lines output should be only taken from the student answer. Do not write anything other than that. If the answer is empty, output 0.1, and if there is no miniscule relation between the answer and question, output 0.0. There should be no preamble or explanation."
                    " \n{system_message}",
                ),
            ]
        )
        prompt = prompt.partial(system_message=system_message)
        return prompt | llm | JsonOutputParser()

    def create_grader_agent(llm, system_message: str):
        """Create an agent."""
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Please act as an impartial judge and evaluate the quality of the candidate's responses to the user's questions below, focusing on correctness and helpfulness. For each question, you will be provided with a reference answer and the assistant's answer. Perform the following steps for each question: 1. Compare the candidate's answer with the reference answer, identifying any mistakes or omissions. 2. Provide an objective explanation highlighting the differences and corrections. 3. Rate the candidate's response on a scale from 1 to 10. The input will be in the format 'Question#. Question: Reference Answer, Student Answer'. YOU MUST Output your evaluation for each question in the following JSON format: {{\"question\": \"{{the question}}\", \"explanation\": \"{{your explanation}}\", \"rating\": \"{{rating}}\"}},..."
                    " \n{system_message}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        prompt = prompt.partial(system_message=system_message)
        return prompt | llm

    def create_reviewer_agent(llm, system_message: str):
        """Create an agent."""
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Your role is to review the rating and feedback given by the grader, and ensure that all information is correct and factual. The information in the reasoning should primarily be built from the rubric, and the grader's score and reasoning respectively.  \n --- \n The rubric items are formatted in the form 'Question #, question, rubric, answer, grade'. You will be given this item, and also context related to the rubric item from the database we have. \n --- \n Read the reasoning carefully to make sure no hallucination and distraction is there. If you think there is a mistake in the grading regarding the points given, object. Think step by step and review the grading and reasoning for the rubric item in the messages, and make your review concise. If there is no mistake in the grade of a rubric item, start your review with 'FINAL POINTS:', otherwise start with 'WRONG POINTS:', and you must start with either. The conversation state will contains the grades in the format 'score, reasoning', so if the score is correct, do not output 'WRONG POINTS:'. If you think the grader gave the correct points, just make sure mentions what the rubric expected. The beginning of the review is only two options: 'FINAL POINTS:' if the grade gave the correct points, and 'WRONG POINTS:' if the grade did not give the correct points"
                    " \n{system_message}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        prompt = prompt.partial(system_message=system_message)
        return prompt | llm
    #Helper function to create a node for AI detector agent
    def detector_node(state, agent, name, items):
        total_score = 0.0
        i = 0
        lines = []
        for question in questions:
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

    # Helper function to create a node for both grader and reviewer agents
    def agent_node(state, agent, name, questions):
        global questions_and_rubric, context
        messages = state["messages"]
        q_a_pairs = ""
        answers = []
        for i, question in enumerate(questions):
            answers.append([])
            for j, subquestion in enumerate(question):
                answers[-1].append([])
                prev_q = None
                for k, string in enumerate(subquestion):
                    if answers[-1][-1]:
                        q_a_pair = format_qa_pair(prev_q,answers[-1][-1])
                        q_a_pairs = q_a_pairs + "\n---\n"+  q_a_pair
                    if messages:
                        # logging.debug(f"Received human: {[HumanMessage(content=string)] + [messages[-len(questions)+i][j][k]]}")
                        if name == "Grader":
                            # get the last grade and review 
                            if ensemble_retriever:
                                current_state = {
                                        "messages": [HumanMessage(content=string)] + [messages[-len(questions)+i][j][k]],
                                        "sender": name,
                                        "q_a_pairs": q_a_pairs,
                                        "context": context[i][j][k]
                                }
                            else:
                                current_state = {
                                    "messages": [HumanMessage(content=string)] + [messages[-len(questions)+i][j][k]],
                                    "sender": name,
                                    "q_a_pairs": q_a_pairs,
                                }
                        # get the last grade given to review 
                        else:
                            if ensemble_retriever: 
                                current_state = {
                                        "messages": [HumanMessage(content=string)] + [messages[-len(questions)+i][j][k]],
                                        "sender": name,
                                        "q_a_pairs": q_a_pairs,
                                        "context": context[i][j][k]
                                }
                            else:
                                current_state = {
                                    "messages": [HumanMessage(content=string)] + [messages[-len(questions)+i][j][k]],
                                    "sender": name,
                                    "q_a_pairs": q_a_pairs,
                                }
                    else:
                        # logging.debug(f"Received human: {[HumanMessage(content=string)]}")
                        if ensemble_retriever: 
                            current_state = {
                                "messages": [HumanMessage(content=string)],
                                "sender": name,
                                "q_a_pairs": q_a_pairs,
                                "context": context[i][j][k]
                            }
                        else:
                            current_state = {
                                "messages": [HumanMessage(content=string)],
                                "sender": name,
                                "q_a_pairs": q_a_pairs,
                            }
                    prev_q = question
                    result = agent.invoke(current_state)
                    answers[-1][-1].append(result.content)
        # We convert the agent output into a format that is suitable to append to the global state
        # all_answers = "\n".join(answers)
        # result = AIMessage(content=all_answers, **result.dict(exclude={"content", "type", "name"}), name=name)
        # result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
        if name == "Reviewer":
            return {
                "messages": [[[message + " Review:" + answer for message,answer in zip(subRow1, subRow2)] for subRow1, subRow2 in zip(row1, row2)] for row1, row2 in zip(messages[-len(answers):], answers)],
                "sender": name,
            }
        if name == "Grader":
            return {
                "messages": answers,
                "sender": name,
            }
    # AI Detector agent and node
    detector_agent = create_detector_agent(
        llm,
        system_message="You should determine whether there is AI-content in the student answers with a score from [0.0 - 100.0], which is the magnitude of AI-content generation. In the lines you output for the AI-generation, make sure those lines are actually in the student answer and no hallucination is there. If you don't think there is AI-generated content, do not add anything to the lines.",
    )
    detector_node = functools.partial(detector_node, agent=detector_agent, name="Detector", items=qa)

    # Grader agent and node
    grader_agent = create_grader_agent(
        llm,
        system_message="You should grade the student answer, given the context and reference answer, and give feedback to the student using that information. You must also give a rating for the student's answer from 1 to 10."
    )
    grader_node = functools.partial(agent_node, agent=grader_agent, name="Grader", questions=qra)

    # Reviewer agent and node
    review_agent = create_reviewer_agent(
        llm,
        system_message="You should make sure the grader follows the rubric primarily. Do not go against the rubric information and assume anything on your own. If the answer satisfies the rubric, do not give a reason to not give the point. Only follow the current rubric item. Other rubric items should not affect your judgement.Do not assume typos, go with what is given to you. If the points are rewarded, do not mention anything in the explanation, except the fact that it satisfied whatever is on the rubric. For negative rubric points, treat it as a binary option between 0 and the negative value, so if the rubric condition is true, then give it the negative points, else if the rubric requirement is not satisfied, give it 0 if there are negative points. If the points rewarded align, then make sure to start with 'FINAL POINTS:', else start with 'WRONG POINTS:' 'WRONG POINTS:' is given only if the score given by you is not the same as the score given by the grader, do not misuse it."
    )
    reviewer_node = functools.partial(agent_node, agent=review_agent, name="Reviewer", questions=qra)

    def router(state):
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
            if not any("WRONG POINTS" in item for sublist in messages[-len(questions):] for sublist2 in sublist for item in sublist2) and state["sender"] == "Reviewer":
                # Only the specified agent is allowed to end the process
                return END

            return "continue"
    workflow = StateGraph(AgentState)
    # workflow.add_node("Detector", detector_node)
    workflow.add_node("Grader", grader_node)
    workflow.add_node("Reviewer", reviewer_node)

    # workflow.add_conditional_edges(
    #     "Detector",
    #     router,
    #     {"continue": "Grader", END: END},
    # )

    workflow.add_conditional_edges(
        "Grader",
        router,
        {"continue": "Reviewer", END: END},
    )

    workflow.add_conditional_edges(
        "Reviewer",
        router,
        {"continue": "Grader", END: END},
    )

    # workflow.add_edge(START, "Detector")
    workflow.add_edge(START, "Grader")
    graph = workflow.compile()
    events = graph.stream(
        {
            "messages": [
            ],
        },
        # Maximum number of steps to take in the graph
        {"recursion_limit": 10},
    )
    last_response = None
    try:
        # print(events)
        for s in events:
            # print(s)
            print("----")
            if "Grader" in s:
                last_response = s
    except Exception as e:
        print(e)
        print(f"final grade")
        logging.debug(f"Received error: ERRORRR") 
    logging.debug(f"Received grade: {last_response["Grader"]['messages']}")
    grade = last_response["Grader"]['messages']
    final = [score[0][0] for score in grade]
    logging.debug(f"Received final: {final}")
    json_objects = []
    for item in final:
        print(item)
        # Replace escaped single quotes with regular single quotes
        item = item.replace("\\'", "'")
        
        # Now try loading the item using json.loads
        try:
            json_objects.append(json.loads(item))
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            print("Problematic string:", item)
    # Write to an actual JSON file
    output_filename = "output.json"
    with open(output_filename, "w") as json_file:
        json.dump(json_objects, json_file, indent=4)
    # score = []
    # summarize_grader_template = """You are a helpful assistant that gets feedback from a grader that will be given to a student for their answers, and the feedback will be right after each other in a single string. I want you to create an informative summary of the feedback that will still convey the main information on what they did right or wrong, but I just want it smaller and less redundant.  Here is the entire feedback list {feedback}. The output should be 'feedback summary'. Do not output anything other than the summary, with no exception.
    #     Output (feedback summary):"""
    # get_summary = ChatPromptTemplate.from_template(summarize_grader_template)
    # llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    # generate_summary = (get_summary | llm | StrOutputParser())
    # final_score = 0
    # for i in range(len(grade)):
    #     for j in range(len(grade[i])):
    #         total = 0
    #         string = ""
    #         for k in range(len(grade[i][j])):
    #             # match = re.search(r'score:\s*(\d+)(?=:|$)', grade[i][j])
    #             match = re.search(r'score:\s*(\d+):\s*(.*)', grade[i][j][k])
    #             if match:
    #                 total += int(match.group(1))  
    #                 string = string + match.group(2) + " "
    #             else:
    #                 total += 0
    #         s = str(i+1) + chr(64 + j + 1) +  ". " + "Score: " + str(total) + " " + generate_summary.invoke({"feedback":string})
    #         score.append(s)
    #         final_score += total
    # print(score)
    # feedback = generate_summary.invoke({"feedback":combined_text})
    return jsonify({"feedback": final}), 200
if __name__ == '__main__':
    app.run(debug=True, port=5000)
