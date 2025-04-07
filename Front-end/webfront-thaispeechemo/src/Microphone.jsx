import React, { useRef, useState } from "react";
import "./App.css";

function Microphone() {
  const mediaRecorderRef = useRef(null);
  const [audioBlob, setAudioBlob] = useState(null);
  const [recording, setRecording] = useState(false);
  const [result, setResult] = useState("");
  const [error, setError] = useState("");

  const handleMicClick = async () => {
    if (recording) {
      mediaRecorderRef.current.stop();
      setRecording(false);
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      let chunks = [];

      mediaRecorder.ondataavailable = (e) => {
        chunks.push(e.data);
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunks, { type: "audio/webm" });
        setAudioBlob(blob);
        chunks = [];
      };

      mediaRecorder.start();
      mediaRecorderRef.current = mediaRecorder;
      setRecording(true);
      setError("");
    } catch (err) {
      setError("⚠️ Cannot access microphone or permission denied.");
    }
  };

  const handleCheck = async () => {
    if (!audioBlob) {
      setError("Please record something first.");
      return;
    }

    const formData = new FormData();
    formData.append("audio", audioBlob, "recording.webm");

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

  const handleClear = () => {
    setAudioBlob(null);
    setResult("");
    setError("");
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
      <h2>Voice Microphone</h2>

      <div onClick={handleMicClick} style={{ cursor: "pointer" }}>
        <img
          src="https://img.icons8.com/ios-filled/100/ffffff/microphone.png"
          alt="mic"
          style={{
            backgroundColor: recording ? "#e04c4c" : "#2c6e7f",
            borderRadius: "50%",
            padding: 20,
            transition: "background-color 0.2s ease",
          }}
        />
      </div>

      <button className="check-btn" onClick={handleCheck}>
        check
      </button>

      {audioBlob && (
        <div style={{ marginTop: 20 }}>
          <p style={{ fontWeight: "bold" }}>🔊 Preview your voice:</p>
          <audio controls src={URL.createObjectURL(audioBlob)} />
          <div>
            <button onClick={handleClear} style={{ marginTop: 10 }}>
              Clear recording
            </button>
          </div>
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

export default Microphone;
