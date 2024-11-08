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

app = Flask(__name__)
app.config.from_object(Config)
main_access = Global()


@app.route('/add-web-content', methods=['POST'])
def add_web_content():
    """Add new web content to the Retriever and update splits."""
    data = request.json
    web_paths = data.get('web_paths')  # Extract 'web_paths' from JSON
    course_id = data.get("course_id")
    course = main_access.courses[course_id]

    if not web_paths or not isinstance(web_paths, list):
        return jsonify({"error": "Please provide a list of URLs in 'web_paths'."}), 400

    # Add documents and update splits
    result = course.retriever.add_documents(web_paths)

    if isinstance(result, int):
        return jsonify({"message": f"{result} new documents added."}), 200
    else:
        return jsonify({"error": result}), 500
    
# Routes for handling input assignments PDFs
@app.route('/upload', methods=['POST'])
def upload_assignment_pdf():
    """Endpoint to upload and convert PDF assignments to LaTeX"""
    # add converter and then we need llm agent to parse the file to get it in the format we want
    data = request.json
    name = data.get("name")
    questions = data.get("questions")
    course_id = data.get("course_id")
    rubric = data.get("rubric")
    if name in main_access.courses[course_id].assignments:
        return jsonify({"success": '-1'}), 200 
    assignment = Assignment(app, questions, rubric)
    main_access.courses[course_id].add_assignment(assignment)
    return jsonify({"success": '1'}), 200 
    #TODO
# Routes for indexing course material
@app.route('/index-course-material', methods=['POST'])
def index_course_material():
    """Endpoint to index course material (PDF/Docx)."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request."}), 400
    file = request.files['file']
    # TODO

# Route for AI-generated content detection
@app.route('/detect-ai-content', methods=['POST'])
def detect_ai_content():
    """Detect AI-generated content in the submission"""
    data = request.json.get('text_data')
    # TODO
    return jsonify({"detection_results": 'detection_results'}), 200

# Route for grading and commenting assignments with multi-layer verification
@app.route('/grade-assignment', methods=['POST'])
def grade_assignment_route():
    """Grade and commentan assignment PDF"""
    course_id = request.json.get('course_id')
    assignment_id = request.json.get('assignment_id') 
    # Example of returned structure:
    return jsonify({"grade_and_comment": [
        {'question': 'Q1', 'student_answer': '...', 'grade': 'A', 'feedback': 'Great job!'},
        {'question': 'Q2', 'student_answer': '...', 'grade': 'B', 'feedback': 'Needs improvement.'},
    ]}), 200

@app.route('/create_account', methods=['POST'])
def create_account():
    """Endpoint to create an account for a human"""
    # add converter and then we need llm agent to parse the file to get it in the format we want
    data = request.json
    email = data["email"]
    password = data["password"]
    with open("accounts.json", "r") as file:
        data = json.load(file)
        if email in data:
            return -1
        id = data["id"]
        data[email] = [password, id]
        return 1
    
    #TODO


if __name__ == '__main__':
    app.run(debug=True)
