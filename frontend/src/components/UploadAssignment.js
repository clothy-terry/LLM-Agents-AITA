import React, { useState } from 'react';
import axios from 'axios';

function UploadAssignment() {
  const [file, setFile] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://localhost:5000/upload', formData);
      alert('Assignment uploaded successfully');
    } catch (error) {
      alert('Error uploading assignment: ' + error.response.data.error);
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <h2>Upload Assignment</h2>
      <input type="file" accept=".pdf" onChange={(e) => setFile(e.target.files[0])} />
      <button type="submit">Upload Assignment</button>
    </form>
  );
}

export default UploadAssignment;