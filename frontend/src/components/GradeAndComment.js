import React, { useState } from 'react';
import axios from 'axios';
import { Button, Typography, Box } from '@mui/material';

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
    <Box>
      
      <Button variant="contained" color="secondary" onClick={handleSubmit}>
        Grade and Comment
      </Button>
      <Box mt={2}>
        <Typography variant="h6">Results:</Typography>
        {result.map((res, index) => (
          <Box key={index} mb={1}>
            <Typography>Question: {res}</Typography>
          </Box>
        ))}
        {grade !== null && <Typography>Grade: {grade}</Typography>}
      </Box>
    </Box>
  );
}

export default GradeAndComment;