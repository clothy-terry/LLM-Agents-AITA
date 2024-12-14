import React from "react";
import { Container, Typography, Box, Button } from "@mui/material";
import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom"; // Import Routes and Route
import UploadCourseMaterial from "./components/UploadCourseMaterial";
import UploadAssignment from "./components/UploadAssignment";
import GradeAndComment from "./components/GradeAndComment";
import UploadAnswers from "./components/UploadAnswers";
import UploadRubric from "./components/UploadRubric";
import AddWebPath from "./components/AddWebPath";
import AnswerQuestions from "./components/AnswerQuestions";

function App() {
  return (
    <Router>
      <Routes>
        {/* Define a route for the main page */}
        <Route path="/" element={<MainPage />} />
        
        {/* Define a route for the answering questions page */}
        <Route path="/answer-questions-page" element={<AnswerQuestionsPage />} />
      </Routes>
    </Router>
  );
}

function MainPage() {
  return (
    <Box style={{ backgroundColor: '#f0f8ff', minHeight: '100vh', padding: '20px' }}>
      <Box display="flex" justifyContent="center" mb={2}>
        <Typography variant="h3" component="h1" gutterBottom style={{ color: '#4b0082' }}>
          Autograder
        </Typography>
      </Box>
      <Container maxWidth="md" style={{ backgroundColor: 'white', padding: '20px', borderRadius: '8px' }}>
        <Box mt={4} display="flex" justifyContent="center">
          <Link to="/answer-questions-page">
            <Button variant="contained" color="primary">
              Ask Question
            </Button>
          </Link>
        </Box>
        <Box mb={2}>
          <UploadCourseMaterial />
        </Box>
        <Box mb={2}>
          <AddWebPath />
        </Box>
        <Box mb={2}>
          <UploadAssignment />
        </Box>
        <Box mb={2}>
          <UploadRubric />
        </Box>
        <Box mb={2}>
          <UploadAnswers />
        </Box>
        <Box mb={2}>
          <GradeAndComment />
        </Box>
      </Container>
    </Box>
  );
}

function AnswerQuestionsPage() {
  return (
    <Box style={{ backgroundColor: '#f0f8ff', minHeight: '100vh', padding: '20px' }}>
      <Box display="flex" justifyContent="center" mb={2}>
        <Typography variant="h3" component="h1" gutterBottom style={{ color: '#4b0082' }}>
          Autograder
        </Typography>
      </Box>
      <Container maxWidth="md" style={{ backgroundColor: 'white', padding: '20px', borderRadius: '8px' }}>
        <Box mb={2}>
          <AnswerQuestions />
        </Box>
      </Container>
    </Box>
  );
}

export default App;
