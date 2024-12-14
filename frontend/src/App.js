import React from "react";
import { Container, Typography, Box } from "@mui/material";
import UploadCourseMaterial from "./components/UploadCourseMaterial";
import UploadAssignment from "./components/UploadAssignment";
import GradeAndComment from "./components/GradeAndComment";
import UploadAnswers from "./components/UploadAnswers";
import UploadRubric from "./components/UploadRubric";
import AddWebPath from "./components/AddWebPath";

function App() {
  return (
    <Box style={{ backgroundColor: '#f0f8ff', minHeight: '100vh', padding: '20px' }}>
      <Box display="flex" justifyContent="center" mb={2}>
        <Typography variant="h3" component="h1" gutterBottom style={{ color: '#4b0082' }}>
          Autograder
        </Typography>
      </Box>
      <Container maxWidth="md" style={{ backgroundColor: 'white', padding: '20px', borderRadius: '8px' }}>
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

export default App;