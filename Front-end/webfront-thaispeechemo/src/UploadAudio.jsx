import React, { useState } from "react";
import { FaCloudUploadAlt } from "react-icons/fa";
import "./App.css";

function UploadAudio() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState("");
  const [error, setError] = useState("");

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (selected && selected.type === "audio/flac") {
      setFile(selected);
      setError("");
      setResult("");
    } else {
      setFile(null);
      setError("Please upload only `.flac` files.");
    }
  };

  const handleCheck = async () => {
    if (!file) {
      setError("No file selected yet!");
      return;
    }

    const formData = new FormData();
    formData.append("audio", file);

    try {
      const res = await fetch("http://localhost:5000/predict-audio", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();

      if (res.ok) {
        setResult(data.emotion);
        setError("");
      } else {
        setError("Server returned an invalid response.");
      }
    } catch (err) {
      setError("Error connecting to the backend.");
    }
  };

  const getEmotionClass = (emotion) => {
    switch (emotion.toLowerCase()) {
      case "neutral":
        return "emotion-neutral";
      case "anger":
      case "angry":
        return "emotion-anger";
      case "happiness":
        return "emotion-happiness";
      case "sadness":
        return "emotion-sadness";
      case "frustration":
        return "emotion-frustration";
      default:
        return "";
    }
  };

  return (
    <div className="container">
      <h2>Upload Audio</h2>

      <div className="upload-container">
        <label className="upload-box-large">
          <FaCloudUploadAlt style={{ marginRight: 10 }} />
          Upload Audio
          <input
            type="file"
            accept=".flac"
            onChange={handleFileChange}
            hidden
          />
        </label>

        <button className="check-btn" onClick={handleCheck}>
          check
        </button>
      </div>

      {file && (
        <div style={{ marginTop: 20 }}>
          <p style={{ fontWeight: "bold" }}>🔊 Preview uploaded file:</p>
          <audio controls src={URL.createObjectURL(file)} />
        </div>
      )}

      {error && <p style={{ color: "red", marginTop: 20 }}>{error}</p>}

      <div style={{ marginTop: "40px", textAlign: "left" }}>
        <h3>Result</h3>
        {result && (
          <>
            <p>From sound is</p>
            <div className={`result-box ${getEmotionClass(result)}`}>
              {result}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

export default UploadAudio;
