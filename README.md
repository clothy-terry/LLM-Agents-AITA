# LLM-Agents-AITA

A system that uses LLM-powered agents to detect AI-generated content and grade assignments with multi-layer verification.

## Project Structure

### `backend.py`
Flask backend application that provides REST API endpoints for:
- PDF assignment upload and conversion
- AI-generated content detection
- Assignment grading with multi-layer verification
- Course content retrieval for RAG (Retrieval-Augmented Generation)
- Reference generation(TODO)
- Math analysis(TODO)
- More...
*Note: Try to keep the backend as simple as possible, and move as much logic as possible to the utils.py file.*

### `rag.ipynb`
Jupyter notebook containing the implementation of:
- LLM-powered multi agent system
- RAG (Retrieval-Augmented Generation) functionality

### `utils.py`
Utility functions for the backend operations:
- PDF text and image extraction
- AI-generated content detection
- Assignment grading logic
- Multi-layer response verification
- Content retrieval for RAG

### `test.py`
Test suite for the application (Currently empty, to be implemented):
- Unit tests for utility functions