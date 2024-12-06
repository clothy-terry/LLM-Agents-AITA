import React, { useState } from 'react';
import axios from 'axios';

function UploadRubric() {
  const [file, setFile] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    const formData = new FormData();
    formData.append('file', file);
    formData.append('course_id', '123');  // Example course_id

    try {
      const response = await axios.post('http://localhost:5000/upload_rubric', formData);
      alert(response.data.message);
    } catch (error) {
      alert('Error uploading rubric: ' + error.response.data.error);
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <h2>Upload Rubric</h2>
      <input type="file" accept=".pdf" onChange={(e) => setFile(e.target.files[0])} />
      <button type="submit">Upload Rubric</button>
    </form>
  );
}

export default UploadRubric;