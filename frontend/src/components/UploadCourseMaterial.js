import React, { useState } from 'react';
import axios from 'axios';

function UploadCourseMaterial() {
  const [file, setFile] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://localhost:5000/upload', formData);
      alert(response.data.message);
    } catch (error) {
      alert('Error uploading course material: ' + error.response.data.error);
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <h2>Upload Course Material</h2>
      <input type="file" onChange={(e) => setFile(e.target.files[0])} />
      <button type="submit">Upload Course Material</button>
    </form>
  );
}

export default UploadCourseMaterial;
