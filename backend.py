from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from utils import (
    functions
)
import os
from retriever import Retriever
from config import Config
from assignment import Assignment

app = Flask(__name__)
app.config.from_object(Config)
retriever = Retriever()

@app.route('/add-web-content', methods=['POST'])
def add_web_content():
    """Add new web content to the Retriever and update splits."""
    data = request.json
    web_paths = data.get('web_paths')  # Extract 'web_paths' from JSON

    if not web_paths or not isinstance(web_paths, list):
        return jsonify({"error": "Please provide a list of URLs in 'web_paths'."}), 400

    # Add documents and update splits
    result = retriever.add_documents(web_paths)

    if isinstance(result, int):
        return jsonify({"message": f"{result} new documents added."}), 200
    else:
        return jsonify({"error": result}), 500
    
# Routes for handling input PDFs and grading
@app.route('/upload', methods=['POST'])
def upload_pdf():
    """Endpoint to upload and convert PDF assignments to LaTeX"""
    # add converter and then we need llm agent to parse the file to get it in the format we want
    questions = None
    assigment = Assignment(app, questions)
    #TODO

# Route for AI-generated content detection
@app.route('/detect-ai-content', methods=['POST'])
def detect_ai_content():
    """Detect AI-generated content in the submission"""
    data = request.json.get('text_data')
    # TODO
    return jsonify({"detection_results": 'detection_results'}), 200

# Route for grading assignments with multi-layer verification
@app.route('/grade-assignment', methods=['POST'])
def grade_assignment_route():
    """Grade an assignment PDF using LLM and multi-layer agents for accuracy and hallucination reduction"""
    

# Route for retrieving relevant course content for RAG purposes
@app.route('/retrieve-course-content', methods=['POST'])
def retrieve_content():
    """Retrieve and provide relevant course content to guide the LLM for grading"""
    # TODO
    return jsonify({"content": 'content'}), 200




if __name__ == '__main__':
    app.run(debug=True)
