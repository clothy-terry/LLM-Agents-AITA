import React, { useState } from 'react';
import axios from 'axios';
//npm install axios


function GradeAndComment() {
  const [result, setResult] = useState([]);
  const [grade, setGrade] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post('http://localhost:5000/grade_assignment', {
        course_id: '123',  // Example course_id
        assignment_id: '456',  // Example assignment_id
      });
      const { feedback, totalGrade } = response.data; 
      setResult(feedback);
      setGrade(totalGrade);
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
            <p>Question: {res}</p>
          </div>
        ))}
        {grade !== null && <p>Grade: {grade}</p>} {/* Display grade if available */}
      </div>
    </div>
  );
}

export default GradeAndComment;
