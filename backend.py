from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from utils import (
    functions
)
import os
from retriever import GradingRetriever
from config import Config
from assignment import Assignment
from human import Human
import json
from main_global import Global
import bs4
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate

from langchain.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
import functools

from langchain_core.messages import AIMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langgraph.graph import END, StateGraph, START
import operator
from typing import Annotated, Sequence
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI

from langchain_community.document_loaders import PyPDFLoader

app = Flask(__name__)
app.config.from_object(Config)
documents = []
splits = []
ensemble_retriever = None 
questions = None
ALLOWED_EXTENSIONS = {'pdf'}
questions_and_rubric = []

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
@app.route('/upload', methods=['POST'])
def upload_assignment_pdf():
    """Endpoint to upload and convert PDF assignments to LaTeX"""
    global questions
    # add converter and then we need llm agent to parse the file to get it in the format we want
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed, only PDFs are allowed'}), 400
    
    # Save the file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    loader = PyPDFLoader(file_path)
    pages = []
    for page in loader.alazy_load():
        pages.append(page)
    combined_text = "\n".join(pages)
    question_answer_template = """You are a helpful assistant that gets the questions from a string of questions, based on the question number and subquestion letter. The questions are formatted with the number, and then the question. I just want the question. Each question/subquestion should separated by '\n'. Here is the entire question list {question}. The output should be 'question#: question'.
        Output (n question # - question pairs):"""
    get_questions = ChatPromptTemplate.from_template(question_answer_template)
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    generate_questions = (get_questions | llm | StrOutputParser() | (lambda x: x.split("\n")))

    questions = generate_questions.invoke({"question":combined_text})
    return jsonify({'message': 'File uploaded successfully'}), 200

# Routes for handling input PDFs for RAG
@app.route('/upload', methods=['POST'])
def upload_pdf():
    """Endpoint to upload and convert PDF assignments to LaTeX"""
    # add converter and then we need llm agent to parse the file to get it in the format we want
    global documents, splits, ensemble_retriever
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed, only PDFs are allowed'}), 400
    
    # Save the file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    loader = PyPDFLoader(file_path)
    try:
        documents = loader.load()
        # Step 2: Split the Text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,  # Number of characters per chunk
            chunk_overlap=50  # Overlap between chunks
        )
        doc_splits = text_splitter.split_documents(documents)

        pages = []
        for page in loader.alazy_load():
            pages.append(page)
        combined_text = "\n".join(pages)
        question_answer_template = """You are a helpful assistant that gets the questions from a string of questions, based on the question number and subquestion letter. The questions are formatted with the number, and then the question. I just want the question. Each question/subquestion should separated by '\n'. Here is the entire question list {question}. The output should be 'question#: question'.
            Output (n question # - question pairs):"""
        get_questions = ChatPromptTemplate.from_template(question_answer_template)
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
        generate_questions = (get_questions | llm | StrOutputParser() | (lambda x: x.split("\n")))

        questions = generate_questions.invoke({"question":combined_text})
        
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

@app.route('/combine_q_r', methods=['POST'])
def combine_q_r():
    global questions_and_rubric

    data = request.json
    rubrics = data.get('rubrics')
    questions_and_rubric = []
    for i in range(len(questions)):
        questions_and_rubric.append([])
        if not questions[i]['subfields']:
            for j in range(len(rubrics[i])):
                questions_and_rubric[-1].append(questions[i] + " - " + rubrics[i][j])
        else:
            for j in range(len(questions[i]['subfields'])):
                questions_and_rubric[-1].append([])
                for k in range(len(rubrics[i][j])):
                    questions_and_rubric[-1][-1].append(questions[i]['subfields'][j] + " - " + rubrics[i][j][k])
    return jsonify({'message': 'Combined questions and rubric successfully'}), 200

# # Route for AI-generated content detection
# @app.route('/detect-ai-content', methods=['POST'])
# def detect_ai_content():
#     """Detect AI-generated content in the submission"""
#     data = request.json.get('text_data')
#     # TODO
#     return jsonify({"detection_results": 'detection_results'}), 200

# Routes for indexing course material
@app.route('/index-course-material', methods=['POST'])
def index_course_material():
    """Endpoint to index course material (PDF/Docx)."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request."}), 400
    file = request.files['file']
    # TODO

# Route for grading and commenting assignments with multi-layer verification
@app.route('/grade-assignment', methods=['POST'])
def grade_assignment_route():
    """Grade an assignment PDF using LLM and multi-layer agents for accuracy and hallucination reduction"""
    data = request.json
    answers = data.get("answers")
    qra = []
    for i in range(len(answers)):
        qra.append([])
        for j in range(questions_and_rubric[i]):
            qra[-1].append(questions_and_rubric[i][j] + " :  " + answers[i])
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
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
                    "You are assuming the role of a student answer grader. You will be given a review of your grading, unless this is the first iteration of grading the answer. If the review exists, and if it starts with 'FINAL GRADE:', then it thinks your grading for that specific rubric item is correct, else it has some improvements that you can take into account. If you think the review improvement advice is not correct, do not follow it, but keep in mind, the reviewer is trying to help, and take its advice seriously. Here are the items that you need to grade based on the question, rubric and answer given. The rubric items are formatted in the form 'Question #, question, rubric, answer'. You will be given this item, plus the previous rubric items+grading scores, and also context related to the rubric item. \n --- \n  You are an agent that primarily uses the rubric item to grade the answer for the provided rubric item. \n The rubric item is provided to you where the points provided corresponds to if the rubric item is true in the student answer. That means the points in the rubric item, no matter if positive or negative, are given only if the rubric item is TRUE in the student answer. If the points is negative, and the rubric item is not satisfied, then give a score of 0. Your final output should be in the format 'score: reasoning' and make sure the reasoning is succinct and to the point. The reasoning should also be focused on the current rubric item only, and it should be directed to the student in the proper tense. \n First, only use the rubric item to give the score, but if you are not confident, you can also use the above context and any background question + answer pairs to help grade the answer for the provided rubric item, but remember that the rubric item is your first and most reliable source of information. If you are giving the student the points, then don't tell what is wrong with it. Just explain why the student did or did not get the points, don't give unneccesary information, so it is concise. Always use the rubric as final call. Think step by step and grade the student answer using the rubric and review as advice. The rubric is the final decision. Go with the rubric."
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
                    "Your role is to review the points and reasoning given by the grader, and ensure that all information is correct and factual. The information in the reasoning should primarily be built from the rubric, and the grader's score and reasoning respectively.  \n --- \n The rubric items are formatted in the form 'Question #, question, rubric, answer, grade'. You will be given this item, and also context related to the rubric item from the database we have. \n --- \n Read the reasoning carefully to make sure no hallucination and distraction is there. If you think there is a mistake in the grading regarding the points given, object. Think step by step and review the grading and reasoning for the rubric item in the messages, and make your review concise. If there is no mistake in the grade of a rubric item, start your review with 'FINAL POINTS:', otherwise start with 'WRONG POINTS:', and you must start with either. The conversation state will contains the grades in the format 'score, reasoning', so if the score is correct, do not output 'WRONG POINTS:'. If you think the grader gave the correct points, just make sure mentions what the rubric expected. The beginning of the review is only two options: 'FINAL POINTS:' if the grade gave the correct points, and 'WRONG POINTS:' if the grade did not give the correct points"
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
                    # get the last grade and review 
                    if ensemble_retriever:
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
                        }
                # get the last grade given to review 
                else:
                    if ensemble_retriever: 
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
                        }
            else:
                if ensemble_retriever: 
                    current_state = {
                        "messages": [HumanMessage(content=question)] + [messages[-len(questions)+i]],
                        "sender": name,
                        # "q_a_pairs": q_a_pairs,
                        # "context": ensemble_retriever.invoke(q)
                    }
                else:
                    current_state = {
                        "messages": [HumanMessage(content=question)],
                        "sender": name,
                        # "q_a_pairs": q_a_pairs,
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
    detector_node = functools.partial(detector_node, agent=detector_agent, name="Detector", items=qs)

    # Grader agent and node
    grader_agent = create_grader_agent(
        llm,
        system_message="You should grade the student answers based on the rubric to the best of your ability. Do not go against the rubric information and assume anything on your own. Do not assume typos, go with what is given to you. Treat each rubric item as a condition, and negative points should be rewarded if the condition is satisfied. Do not take semantics of the rubric into account. Rubric is the truth. Scores can only be 0 or the points shown in the rubric item. ",
    )
    grader_node = functools.partial(agent_node, agent=grader_agent, name="Grader", questions=questions)

    # Reviewer agent and node
    review_agent = create_reviewer_agent(
        llm,
        system_message="You should make sure the grader follows the rubric primarily. Do not go against the rubric information and assume anything on your own. If the answer satisfies the rubric, do not give a reason to not give the point. Only follow the current rubric item. Other rubric items should not affect your judgement.Do not assume typos, go with what is given to you. If the points are rewarded, do not mention anything in the explanation, except the fact that it satisfied whatever is on the rubric. For negative rubric points, treat it as a binary option between 0 and the negative value, so if the rubric condition is true, then give it the negative points, else if the rubric requirement is not satisfied, give it 0 if there are negative points. If the points rewarded align, then make sure to start with 'FINAL POINTS:', else start with 'WRONG POINTS:' 'WRONG POINTS:' is given only if the score given by you is not the same as the score given by the grader, do not misuse it."
    )
    reviewer_node = functools.partial(agent_node, agent=review_agent, name="Reviewer", questions=questions)

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
            if not "WRONG POINTS" in " ".join(messages[-len(questions):]) and state["sender"] == "Reviewer":
                # Only the specified agent is allowed to end the process
                return END
            return "continue"
    workflow = StateGraph(AgentState)
    workflow.add_node("Detector", detector_node)
    workflow.add_node("Grader", grader_node)
    workflow.add_node("Reviewer", reviewer_node)

    workflow.add_conditional_edges(
        "Detector",
        router,
        {"continue": "Grader", END: END},
    )

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

    workflow.add_edge(START, "Detector")
    graph = workflow.compile()

    events = graph.stream(
        {
            "messages": [
            ],
        },
        # Maximum number of steps to take in the graph
        {"recursion_limit": 10},
    )

    try:
        for s in events:
            print(s)
            print("----")
    except Exception as e:
        print(e)
        print(f"final grade")
    # print(questions)
    return jsonify({"grade": 'grade'}), 200
    


if __name__ == '__main__':
    app.run(debug=True)
