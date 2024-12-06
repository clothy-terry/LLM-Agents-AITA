import React, { useState } from 'react';
import axios from 'axios';

function GradeAndComment() {
  const [result, setResult] = useState([]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post('http://localhost:5000/grade-assignment', {
        answers: ['Answer 1', 'Answer 2'],  // Example answers
      });
      setResult(response.data);  // Assuming the backend returns the result directly
    } catch (error) {
      alert('Error grading assignment: ' + error.response.data.error);
    }
  };

  return (
    <div>
      <h2>Grade and Comment</h2>
      <button onClick={handleSubmit}>Grade and Comment</button>
      <div>
        <h3>Results:</h3>
        {result.map((res, index) => (
          <div key={index}>
            <p>Score: {res.score}</p>
            <p>Lines: {res.lines}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

export default GradeAndComment;