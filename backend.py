from flask import Flask, request, jsonify
import os
from config import Config
import bs4
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
import functools
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.schema import Document
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
import logging
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub

logging.basicConfig(level=logging.DEBUG)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str

@app.route('/answer_questions', methods=['POST'])
def answer_questions():
    """Add new web content to the Retriever and update splits."""
    global ensemble_retriever
    web_search_tool = TavilySearchResults(k=3)
    subquestion_template = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
    The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
    Generate multiple search queries related to: {question} \n
    Output (3 queries):"""
    prompt_decomposition = ChatPromptTemplate.from_template(subquestion_template)
    prompt_rag = hub.pull("rlm/rag-prompt")
    # LLM
    llm = ChatOpenAI(temperature=0)
    # Chain
    generate_queries_decomposition = ( prompt_decomposition | llm | StrOutputParser() | (lambda x: x.split("\n")))
    data = request.json  # The data is sent as JSON
    question = data.get('material', '')
    sub_questions = generate_queries_decomposition.invoke({"question":question})
    
    retrieval_grader_prompt = PromptTemplate(
        template="""You are a grader assessing relevance
        of a retrieved document to a user question. If the document contains keywords related to the user question,
        grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals.

        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
        Provide the binary score as a JSON with a single key 'score' and no premable or explaination.

        Here is the retrieved document:
        {document}

        Here is the user question:
        {question}
        """,
        input_variables=["question", "document"],
        )
    retrieval_grader = retrieval_grader_prompt | llm | JsonOutputParser()
    # Initialize a list to hold RAG chain results
    rag_results = []
    if not ensemble_retriever:
        answer_question_template = """You are a helpful assistant that generates answers to an input question. \n
        The goal is to answer the question correctly and truthfully. If you do not know the answer confidently, say that you don't know the answer. \n
        Generate answer to: {question} \n
        Output (1 answer):"""
        generate_answers = ChatPromptTemplate.from_template(answer_question_template)
        generate_answers_to_question = ( generate_answers | llm | StrOutputParser() )
        answer = generate_answers_to_question.invoke({"question":question})
        return jsonify({"answer": answer}), 200
    for sub_question in sub_questions:
        filtered_docs = []
        # Retrieve documents for each sub-question
        retrieved_docs = ensemble_retriever.get_relevant_documents(sub_question)
        for d in retrieved_docs:
          score = retrieval_grader.invoke(
              {"question": question, "document": d.page_content}
          )
          grade = score["score"]
          if grade.lower() == "yes":
              # print("RELEVANT DOC")
              filtered_docs.append(d)
          else:
              # print("NOT RELEVANT")
              web_search = "Yes"
              continue
        if web_search == "Yes":
          docs = web_search_tool.invoke({"query": question})
          web_results = "\n".join([d["content"] for d in docs])
          web_results = Document(page_content=web_results)
          filtered_docs.append(web_results)

        # Use retrieved documents and sub-question in RAG chain
        answer = (prompt_rag | llm | StrOutputParser()).invoke({"context": filtered_docs,
                                                                "question": sub_question})
        rag_results.append(answer)
        result = ["Subquestion: " + question + " | Answer: " + answer for question, answer in zip(sub_questions, rag_results)]
    return jsonify({"answer": result}), 200
    
@app.route('/add-web-content', methods=['POST'])
def add_web_content():
    """Add new web content to the Retriever and update splits."""
    global documents, splits, ensemble_retriever
    try:
        # Get the text data from the request
        data = request.json  # The data is sent as JSON
        material = data.get('material', '')  # Extract the material (URLs) from the request
        urls = re.split(r'[\n,;]+', material)
        web_paths = [url.strip() for url in urls if url.strip()] 
        # For demonstration, you can print or process the URLs here
        print(f"Received URLs: {urls}")

    except Exception as e:
        return jsonify({'error': str(e)}), 400
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
        return jsonify({"error": f"Error loading documents: {str(e)}"}), 500

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
    question_answer_template = """You are a helpful assistant that gets the questions from a string of questions, based on the question number and subquestion letter. The questions are formatted with the number, and if there are subqestions, then the letter would be there, and finally the question. I just want the question. Each question/subquestion should separated by '\n'. Here is the entire question list {question}. The output should be 'question/subquestion#: question'.
        Output (n question #: - question pairs):"""
    get_questions = ChatPromptTemplate.from_template(question_answer_template)
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    generate_questions = (get_questions | llm | StrOutputParser() | (lambda x: x.split("\n")))
    # qs = generate_questions.invoke({"question":combined_text})
    questions = []
    for page in loader.lazy_load():
        # combined_text = page.page_content
        pages.append(page.page_content)
    combined_text = "\n".join(pages)
    qs = generate_questions.invoke({"question":combined_text})
    # questions.append([])
    i = 0
    prev = None
    for question in qs: 
        if question.strip():
            i += 1
            if int(question.strip()[0]) != prev:
                prev = int(question.strip()[0])
                questions.append([])
                questions[-1].append(question.strip())
            else:
                questions[-1].append(question.strip())
    num = i
    logging.debug(f"Received Data: {questions}") 
    return jsonify({'message': 'File uploaded successfully'}), 200

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
    # add converter and then we need llm agent to parse the file to get it in the format we want
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed, only PDFs are allowed'}), 400
    if not questions:
        return jsonify({'error': 'First upload questions'}), 400

    # Save the file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    loader = PyPDFLoader(file_path)
    pages = []
    for page in loader.lazy_load():
        pages.append(page.page_content)
    combined_text = "\n".join(pages)
    # rubric_template = """You are a helpful assistant that gets the rubric items, based on the question number and subquestion letter. The rubrics are formatted with the number, and if there are subqestions, then the letter would be there, and finally the rubric items which contains the quantity of points and condition for the points in the respective order. I just want the rubric items. Each question/subquestion should separated by '\n'. Here is the entire rubric list {rubric}. The output should simply be the rubric items for each subquestion, and do not include the question and subquestion# in the output. Remember to include the number of points which is '+ number' or '- number', and the condition, like 'condition: points'. I WANT all rubric items for the same subquestion to be in the same string. Pretty much, separate the rubric based on subquestions, using '\n'.
    #     Output (n rubric items):"""
    rubric_template = """You are a helpful assistant that gets the rubric items, based on the question number and subquestion letter. The rubrics are formatted with the number, and if there are subqestions, then the letter would be there, and finally the rubric items which contains the quantity of points and condition for the points in the respective order. I just want the rubric items. Each question/subquestion should separated by '\n'. Here is the entire rubric list {rubric}. The output should simply be the rubric items for each subquestion, and do not include the question and subquestion# in the output. Remember to include the number of points which is '+ number' or '- number', and the condition, like 'condition: points'. I WANT all rubric items for the same subquestion to be in the same string. Pretty much, separate the rubric based on subquestions, using '\n'.
        Output (n rubric # - rubric pairs):"""
    get_rubric = ChatPromptTemplate.from_template(rubric_template)
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    generate_rubric = (get_rubric | llm | StrOutputParser() | (lambda x: x.split("\n")))
    rubrics = generate_rubric.invoke({"rubric":combined_text})
    r = [s.strip() for s in rubrics]
    if num != len(rubrics):
        return jsonify({'error': f"Unequal number of questions and rubric items: {num}, {len(rubrics)}"}), 400

    # Define the regex pattern to match and remove
    # pattern = r"^\d\(?[a-zA-Z]\)?[.+]*[.:]"  # Number → Letter → Period at the start

    # Process each string
    # rubrics = [re.sub(pattern, '', s, count=1).strip() for s in stripped_strings]

    temp = []
    idx = 0  # Pointer for the 1D list
    for row in questions:
        # Match the length of the current row in two_d
        temp.append(r[idx:idx + len(row)])
        idx += len(row)  # Move the pointer forward
    rubrics = [[split_points(string) for string in question_num] for question_num in temp]
    logging.debug(f"Received Rubric: {rubrics}") 
    combine()
    return jsonify({'message': 'File uploaded successfully'}), 200

@app.route('/upload_answers', methods=['POST'])
def upload_answer_pdf():
    """Endpoint to upload and convert PDF assignments to LaTeX"""
    global questions, answers, num
    # add converter and then we need llm agent to parse the file to get it in the format we want
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed, only PDFs are allowed'}), 400
    if not questions:
        return jsonify({'error': 'Assignment pdf needs to be uploaded'}), 400
    
    # Save the file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    loader = PyPDFLoader(file_path)
    pages = []
    for page in loader.lazy_load():
        pages.append(page.page_content)
    combined_text = "\n".join(pages)
    answer_template = """You are a helpful assistant that gets the answers from a string of answers, based on the question number and subquestion letter. The answers are formatted with the number, and then it says 'Solution:' followed by the answer. I just want you to get the answer for each subquestion. \\
        Make sure to get the entire answer, and do not summarize or take the last part of answer. I just want the entire answer. You can determine the number of questions from the question list. Here is the question list {question}. Just use the question list to determine the number of questions. Here are the student answers {answer}. The output should be 'answer#: answer'. \\
        If you do not see an answer corresponding to a particular question number/subquestion, I still want an answer to that question in the answer list, so just write 'I do not know the answer'. Do not have empty strings in the list between the answers. If there are subanswers, the main question should not have an answer. 
        Output (n answer #- answer pairs):"""
    get_answers = ChatPromptTemplate.from_template(answer_template)
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    generate_answers = (get_answers | llm | StrOutputParser() | (lambda x: x.split("\n")))

    temp = generate_answers.invoke({"answer":combined_text, "question": questions})
    # answers = ["Student Answer: " + s[s.index('.') + 1:] if '.' in s else s for s in answers]
    temp = [s.strip() for s in temp]
    if num != len(temp):
        return jsonify({'error': f"Unequal number of questions and rubric items: {num}, {len(answers)}"}), 400
    it = iter(temp)

    # Reshape the list
    answers = [[next(it) for _ in sublist] for sublist in questions]

    # # Define the regex pattern to match and remove
    # pattern = r"^\d[a-zA-Z]\."  # Number → Letter → Period at the start

    # # Process each string
    # answers = [re.sub(pattern, '', s, count=1).strip() for s in stripped_strings]

    logging.debug(f"Received Answers: {answers}") 
    return jsonify({'message': 'Answers uploaded successfully'}), 200

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
        
        splits.extend(doc_splits)
        result = len(doc_splits)
        faiss_index = FAISS.from_documents(splits, embedding=OpenAIEmbeddings())
        faiss_retriever = faiss_index.as_retriever()

        bm25_retriever = BM25Retriever.from_documents(splits)

        ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever],
                                    weights=[0.4, 0.6])
    except Exception as e:
        return jsonify({"error": f"Error loading documents: {str(e)}"}), 500

    if isinstance(result, int):
        return jsonify({"message": f"{result} new documents added."}), 200
    else:
        return jsonify({"error": result}), 500

def combine():
    global questions_and_rubric, questions, rubrics
    questions_and_rubric = [
        [
            [s1 + " Rubric: " + s2 for s2 in sub3d]
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
    global questions_and_rubric, answers, questions, qra
    qra = [
        [
            [s2 + " Answer: " + s1 for s2 in sub3d]
            for s1, sub3d in zip(sub2d, sub3d_list)
        ]
        for sub2d, sub3d_list in zip(answers, questions_and_rubric)
    ]
    logging.debug(f"Received qra: {qra}") 
    qa = [[s1 + "Answer: " + s2 for s1, s2 in zip(row1, row2)] for row1, row2 in zip(questions, answers)]
    return qa

# Route for grading and commenting assignments with multi-layer verification
@app.route('/grade_assignment', methods=['POST'])
def grade_assignment_route():
    """Grade an assignment PDF using LLM and multi-layer agents for accuracy and hallucination reduction"""
    # Prompt to decompose rubric items into list of elements where each element contains Question #, Question, Rubric, Student Answer; each element is separated based on rubric item"
    global questions_and_rubric, answers, qra, questions
    qa = combine_ra()
    logging.debug(f"Received qa: {qa}") 
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
                    "You are assuming the role of a student answer grader. You will be given a review of your grading, unless this is the first iteration of grading the answer. If the review exists, and if it starts with 'FINAL GRADE:', then it thinks your grading for that specific rubric item is correct, else it has some improvements that you can take into account. If you think the review improvement advice is not correct, do not follow it, but keep in mind, the reviewer is trying to help, and take its advice seriously. Here are the items that you need to grade based on the question, rubric and answer given. The rubric items are formatted in the form 'Question #, question, rubric, answer'. You will be given this item, plus the previous rubric items+grading scores for the same question number, and also context related to the rubric item. \n --- \n  You are an agent that primarily uses the rubric item to grade the answer for the provided rubric item. \n The rubric item is provided to you where the points provided corresponds to if the rubric item is true in the student answer. That means the points in the rubric item, no matter if positive or negative, are given only if the rubric item is TRUE in the student answer. If the points is negative, and the rubric item is not satisfied, then give a score of 0. Your final output should be in the format 'score: reasoning' and make sure the reasoning is succinct and to the point. The reasoning should also be focused on the current rubric item only, and it should be directed to the student in the proper tense. \n First, only use the rubric item to give the score, but if you are not confident, you can also use the above context and any background question + answer pairs to help grade the answer for the provided rubric item, but remember that the rubric item is your first and most reliable source of information. If you are giving the student the points, then don't tell what is wrong with it. Just explain why the student did or did not get the points, don't give unneccesary information, so it is concise. Always use the rubric as final call. Think step by step and grade the student answer using the rubric and review as advice. The rubric is the final decision. Go with the rubric. Once you are done using the rubric to grade the question and give the number of points, you can use the context to help give feedback and support your decision, but make sure you do not contradict the rubric."
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
    def detecting_node(state, agent, name, items):
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
    def agent_node(state, agent, name, l):
        messages = state["messages"]
        q_a_pairs = ""
        answers = []
        for i, question in enumerate(l):
            answers.append([])
            for j, subquestion in enumerate(question):
                answers[-1].append([])
                prev_q = None
                for k, string in enumerate(subquestion):
                    if answers[-1][-1]:
                        q_a_pair = format_qa_pair(prev_q,answers[-1][-1])
                        q_a_pairs = q_a_pairs + "\n---\n"+  q_a_pair
                    if messages:
                        if name == "Grader":
                            # get the last grade and review 
                            if ensemble_retriever:
                                current_state = {
                                        "messages": [HumanMessage(content=string)] + [messages[-len(l)+i][j][k]],
                                        "sender": name,
                                        "q_a_pairs": q_a_pairs,
                                        "context": ensemble_retriever.invoke(questions[i][j])
                                }
                            else:
                                current_state = {
                                    "messages": [HumanMessage(content=string)] + [messages[-len(l)+i][j][k]],
                                    "sender": name,
                                    "q_a_pairs": q_a_pairs,
                                }
                        # get the last grade given to review 
                        else:
                            if ensemble_retriever: 
                                current_state = {
                                        "messages": [HumanMessage(content=string)] + [messages[-len(l)+i][j][k]],
                                        "sender": name,
                                        "q_a_pairs": q_a_pairs,
                                        "context": ensemble_retriever.invoke(questions[i][j])
                                }
                            else:
                                current_state = {
                                    "messages": [HumanMessage(content=string)] + [messages[-len(l)+i][j][k]],
                                    "sender": name,
                                    "q_a_pairs": q_a_pairs,
                                }
                    else:
                        if ensemble_retriever: 
                            current_state = {
                                "messages": [HumanMessage(content=string)],
                                "sender": name,
                                "q_a_pairs": q_a_pairs,
                                "context": ensemble_retriever.invoke(questions[i][j])
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
    detector_node = functools.partial(detecting_node, agent=detector_agent, name="Detector", items=qa)

    # Grader agent and node
    grader_agent = create_grader_agent(
        llm,
        system_message="You should grade the student answers based on the rubric to the best of your ability. Do not go against the rubric information and assume anything on your own. Do not assume typos, go with what is given to you. Treat each rubric item as a condition, and negative points should be rewarded if the condition is satisfied. Do not take semantics of the rubric into account. Rubric is the truth. Scores can only be 0 or the points shown in the rubric item. ",
    )
    grader_node = functools.partial(agent_node, agent=grader_agent, name="Grader", l=qra)

    # Reviewer agent and node
    review_agent = create_reviewer_agent(
        llm,
        system_message="You should make sure the grader follows the rubric primarily. Do not go against the rubric information and assume anything on your own. If the answer satisfies the rubric, do not give a reason to not give the point. Only follow the current rubric item. Other rubric items should not affect your judgement.Do not assume typos, go with what is given to you. If the points are rewarded, do not mention anything in the explanation, except the fact that it satisfied whatever is on the rubric. For negative rubric points, treat it as a binary option between 0 and the negative value, so if the rubric condition is true, then give it the negative points, else if the rubric requirement is not satisfied, give it 0 if there are negative points. If the points rewarded align, then make sure to start with 'FINAL POINTS:', else start with 'WRONG POINTS:' 'WRONG POINTS:' is given only if the score given by you is not the same as the score given by the grader, do not misuse it."
    )
    reviewer_node = functools.partial(agent_node, agent=review_agent, name="Reviewer", l=qra)

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
    # workflow.add_edge(START, "Grader")
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
    logging.debug(f"Received grade: {last_response['Grader']['messages']}")
    grade = last_response["Grader"]['messages']
    score = []
    summarize_grader_template = """You are a helpful assistant that gets feedback from a grader that will be given to a student for their answers, and the feedback will be right after each other in a single string. I want you to create an informative summary of the feedback that will still convey the main information on what they did right or wrong, but I just want it smaller and less redundant.  Here is the entire feedback list {feedback}. The output should be 'feedback summary'. Do not output anything other than the summary, with no exception.
        Output (feedback summary):"""
    get_summary = ChatPromptTemplate.from_template(summarize_grader_template)
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    generate_summary = (get_summary | llm | StrOutputParser())
    final_score = 0
    for i in range(len(grade)):
        for j in range(len(grade[i])):
            total = 0
            string = ""
            for k in range(len(grade[i][j])):
                # match = re.search(r'score:\s*(\d+)(?=:|$)', grade[i][j])
                match = re.search(r'score:\s*(\d+):\s*(.*)', grade[i][j][k])
                if match:
                    total += int(match.group(1))  
                    string = string + match.group(2) + " "
                else:
                    total += 0
            s = str(i+1) + chr(64 + j + 1) +  ". " + "Score: " + str(total) + " " + generate_summary.invoke({"feedback":string})
            score.append(s)
            final_score += total
    # print(score)
    # feedback = generate_summary.invoke({"feedback":combined_text})
    return jsonify({"feedback": score, "totalGrade": final_score}), 200
if __name__ == '__main__':
    app.run(debug=True, port=5000)
