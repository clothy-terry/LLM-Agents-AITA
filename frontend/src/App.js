import React from 'react';
import UploadCourseMaterial from './components/UploadCourseMaterial';
import UploadAssignment from './components/UploadAssignment';
import GradeAndComment from './components/GradeAndComment';
import UploadAnswers from './components/UploadAnswers';

function App() {
  return (
    <div className="App">
      <h1>AI-Powered Teaching Assistant</h1>
      <UploadCourseMaterial />
      <UploadAssignment />
      <UploadAnswers />
      <GradeAndComment />
    </div>
  );
}

export default App;
