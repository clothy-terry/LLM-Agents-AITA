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

function AddWebPath() {
  // State to store the user's input (text containing one or more URLs)
  const [textInput, setTextInput] = useState("");

  // Function to handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault(); // Prevent the default form submission behavior

    try {
      // Send the input data (textInput) to the server
      const response = await axios.post("http://localhost:5001/add-web-content", {
        material: textInput, // Pass the input as the 'material' field in the request body
      });
      alert(response.data.message); // Alert the user with the server's response message
    } catch (error) {
      // Handle errors and display an alert with a relevant message
      alert("Error uploading urls: " + (error.response?.data?.error || error.message));
    }
  };

  // Function to clear the text input
  const handleClear = () => {
    setTextInput(""); // Reset the state to an empty string
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
          Add Web Paths
        </Typography>
        <Tooltip
          title={
            <Typography variant="body1">
              Enter one or more URLs separated by commas or line breaks.
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
          label="Enter URLs"
          value={textInput} // Bind the input value to state
          onChange={(e) => setTextInput(e.target.value)} // Update state on input change
          style={{ width: "100%", marginBottom: "16px" }}
          margin="normal"
          multiline // Allow multiple lines of text
          rows={4} // Set initial number of rows to 4
          variant="outlined" // Use an outlined style for the TextField
          placeholder="https://example.com, https://example2.com" // Placeholder text
        />
        <Box display="flex" alignItems="center">
          {/* Submit button to send data to the server */}
          <Button
            variant="contained"
            color="primary"
            type="submit"
            style={{ marginRight: "8px" }}
          >
            Add Web Paths
          </Button>
          {/* Clear button to reset the input field */}
          <Button variant="outlined" color="primary" onClick={handleClear}>
            Clear
          </Button>
        </Box>
      </Box>
    </form>
  );
}

export default AddWebPath;
