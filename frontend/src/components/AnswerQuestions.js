import React, { useState } from "react";
import axios from "axios";
import {
  Button,
  TextField,
  Typography,
  Box,
  Tooltip,
  IconButton,
} from "@mui/material";
import InfoIcon from "@mui/icons-material/Info";

function AnswerQuestions() {
  // State to store the user's input
  const [textInput, setTextInput] = useState("");
  // State to store the response from the server
  const [serverResponse, setServerResponse] = useState("");

  // Function to handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault(); // Prevent the default form submission behavior

    try {
      // Send the input data (textInput) to the server
      const response = await axios.post("http://localhost:5000/answer_questions", {
        material: textInput, // Pass the input as the 'material' field in the request body
      });
      // Set the server response in the state
      setServerResponse(response.data.answer || "No response from server");
    } catch (error) {
      // Handle errors and display an alert with a relevant message
      setServerResponse("Error asking question: " + (error.response?.data?.error || error.message));
    }
  };

  // Function to clear the text input and response
  const handleClear = () => {
    setTextInput(""); // Reset the state to an empty string
    setServerResponse(""); // Clear the previous server response
  };

  return (
    <form onSubmit={handleSubmit}>
      <Box display="flex" alignItems="center">
        <Typography
          variant="h5"
          component="h2"
          gutterBottom
          style={{ marginBottom: "-10px" }}
        >
          Answer Question
        </Typography>
        <Tooltip
          title={
            <Typography variant="body1">
              Enter question
            </Typography>
          }
          sx={{ fontSize: "1.2em" }}
        >
          <IconButton>
            <InfoIcon />
          </IconButton>
        </Tooltip>
      </Box>
      <Box display="flex" flexDirection="column" alignItems="start">
        {/* Multiline TextField to accept user input */}
        <TextField
          label="Enter Question"
          value={textInput} // Bind the input value to state
          onChange={(e) => setTextInput(e.target.value)} // Update state on input change
          style={{ width: "100%", marginBottom: "16px" }}
          margin="normal"
          multiline // Allow multiple lines of text
          rows={4} // Set initial number of rows to 4
          variant="outlined" // Use an outlined style for the TextField
          placeholder="What is 1+1" // Placeholder text
        />
        <Box display="flex" alignItems="center">
          {/* Submit button to send data to the server */}
          <Button
            variant="contained"
            color="primary"
            type="submit"
            style={{ marginRight: "8px" }}
          >
            Answer Question
          </Button>
          {/* Clear button to reset the input field */}
          <Button variant="outlined" color="primary" onClick={handleClear}>
            Clear
          </Button>
        </Box>
      </Box>

      {/* Display the server response */}
      {serverResponse && (
        <Box mt={2}>
          <Typography variant="h6">Server Response:</Typography>
          {Array.isArray(serverResponse) ? (
            // Render each item in the list
            serverResponse.map((item, index) => (
              <Typography key={index} variant="body1">
                {item}
              </Typography>
            ))
          ) : (
            // Render the string response
            <Typography variant="body1">{serverResponse}</Typography>
          )}
        </Box>
      )}
    </form>
  );
}

export default AnswerQuestions;
