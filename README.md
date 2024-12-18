# LLM-Agents-AITA

A system using LLM-powered agents to detect AI-generated content and grade assignments with multi-layer verification.

---

## Installation Instructions

### Backend Setup

Run the following in the project **root directory**:

```bash
pip install -r requirements.txt
```

### Frontend Setup

Run the following in the **frontend directory** (cd frontend):

```bash
npm install
npm install @mui/material @emotion/react @emotion/styled
npm install @mui/icons-material
```

---

## Environment Configuration

1. Create a `.env` file in the **root directory** (`/LLM-Agents-AITA`) with the following keys:
   ```plaintext
   LANGCHAIN_API_KEY=your_key
   TAVILY_API_KEY=your_key
   OPENAI_API_KEY=your_key
   ```

---

## Running the Application

### Start the Frontend

```bash
cd frontend
npm start
```

### Start the Backend in Separate Terminal

```bash
python backend.py
```

---

## Demo Instructions

1. Open the "Ask a Question" page and ask a question.
2. Go to main page and upload the following files sequentially, **waiting for a success notification after each upload**:
   - `demo_eval/context.pdf` (Course Material)
   - `demo_eval/assignment.pdf` (Assignment)
   - `demo_eval/rubric.pdf` (Rubric)
   - `demo_eval/answers.pdf` (Answers)
3. Click **"Grade and Comment"**.
4. Monitor progress in the backend server terminal. Grading may take some time.

---

### Fixing Warning/Error

If you encounter the warning or error mentioned above:

> One of your dependencies, `babel-preset-react-app`, is importing the `@babel/plugin-proposal-private-property-in-object` package without declaring it in its dependencies. This is currently working because `@babel/plugin-proposal-private-property-in-object` is already in your `node_modules` folder for unrelated reasons, but it may break at any time.

You can fix it by adding the missing package manually to your `devDependencies`. Run the following command:

```bash
npm install --save-dev @babel/plugin-proposal-private-property-in-object
```

---

## Project Structure

### `frontend`

The frontend of this project is built using React, Material-UI (MUI), and Axios to create a scalable and user-friendly interface.
Key Features include:

- Upload Course Material: Allows users to upload various course-related documents, implemented in the UploadCourseMaterial component.
- Upload Assignment: Allows users to upload assignments, enabling the system to process and prepare them for grading, handled by the UploadAssignment component.
- Upload Rubric: Users can upload grading rubrics, which the AI uses to evaluate student submissions, managed by the UploadRubric component.
- Upload Answers: Allows users to upload student answers for automated grading, implemented in the UploadAnswers component.
- Grade and Comment: Once all necessary documents are uploaded, this feature allows the AI to grade the assignments and provide feedback.

### `backend.py`

Flask backend application that provides REST API endpoints for:

- AI-generated content detection: LLM agent that aims to detect AI-generated content in student answers
- Assignment grading with multi-layer verification: 2 LLM agents conversing with each other to determine best grade to give
- Course content retrieval for RAG (Retrieval-Augmented Generation): Weighted sum of two retrievers to get best documents related to questions
  - PDF assignment upload and conversion: Upload pdf and update retriever
  - Web path upload and conversion: Upload web path and update retriever
- Question-Answering tool using RAG
  - Uses Retrieval Grader and Web search tool to help get context on how to answer the question

### `rag.ipynb`

Jupyter notebook containing the implementation of:

- LLM-powered multi agent system
- RAG (Retrieval-Augmented Generation) functionality

## Evaluation

We plan to create an evaluation dataset with human involvement. The goal is to manually curate a high-quality dataset for various subjects, containing well-crafted QA pairs.

Our approach involves generating this dataset first, then selecting and refining the highest-quality QA pairs. We plan to use a state-of-the-art LLM to parse text from chapters on specific subjects, automatically generating QA pairs. A human will then manually curate the best pairs from this collection.

Depending on the subject matter, we may adopt two different approaches:

### For Subjects Where the Answer to Any Question is Typically Context-Dependent

- Subjects like history, geography, biology, psychology, literature, etc.
- For these subjects, we plan to select a few chapters from typical college-level course material.
- After converting this material to text, we will chunk the relevant sections and feed them into an LLM, such as GPT-4 or Claude, to generate QA pairs from the content.
- We then use human oversight to select the most accurate and relevant QA pairs from those generated by the LLM.
- To make the curation process more efficient and scalable, we intend to explore tools like [Argilla](https://github.com/argilla-io/argilla).

### For Reasoning-Oriented Subjects (e.g., Math)

- For subjects like math where answers are strongly defined, we can use existing question bank datasets
- We need to be careful when using existing datasets, as they may have been used in the model training set
- Datasets like Deepmind's mathematics and MathQA support question generation, so that we're testing on completely new questions (https://github.com/google-deepmind/mathematics_dataset, https://math-qa.github.io/)
- These datasets support questions on various different subfields of mathematics and various levels of difficulty

After curating this dataset, we plan to use an "LLM-as-a-judge" approach to evaluate both our compound system and a baseline (a standard LLM). We aim to follow best practices, as outlined in [this blog](https://hamel.dev/blog/posts/llm-judge/), to ensure that our "LLM-as-a-judge" technique is effective for our scenario.
