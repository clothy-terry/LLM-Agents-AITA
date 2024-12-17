import React, { useState, useRef } from "react";
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

function UploadCourseMaterial() {
  const [file, setFile] = useState(null);
  const fileInputRef = useRef();

  const handleSubmit = async (e) => {
    e.preventDefault();
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post(
        "http://localhost:5001/upload",
        formData
      );
      alert(response.data.message);
    } catch (error) {
      alert("Error uploading course material: " + error.response.data.error);
    }
  };

  const handleClear = () => {
    setFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
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
          Upload Course Material
        </Typography>
        <Tooltip
          title={
            <Typography variant="body1">
              E.g. Lecture slides, course notes, materials relevant to the assignment.
            </Typography>
          }
          sx={{ fontSize: "1.2em" }}
        >
          <IconButton>
            <InfoIcon />
          </IconButton>
        </Tooltip>
      </Box>
      <Box display="flex" alignItems="center">
        <TextField
          type="file"
          onChange={(e) => setFile(e.target.files[0])}
          inputRef={fileInputRef}
          style={{ width: "50%", marginRight: "8px" }}
          margin="normal"
          InputLabelProps={{ shrink: true }}
        />
        <Button
          variant="contained"
          color="primary"
          type="submit"
          style={{ marginRight: "8px" }}
        >
          Upload
        </Button>
        <Button variant="outlined" color="primary" onClick={handleClear}>
          Clear
        </Button>
      </Box>
    </form>
  );
}

export default UploadCourseMaterial;
