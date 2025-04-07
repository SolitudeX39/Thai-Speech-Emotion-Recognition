import React from "react";
import { useNavigate } from "react-router-dom";
import "./App.css";

function Welcome() {
  const navigate = useNavigate();

  return (
    <div className="container">
      <h1 className="title">
        WELCOME to
        <br />
        Thai Speech Emotion Recognition
      </h1>

      <div className="button-container">
        <div className="option-box" onClick={() => navigate("/UploadAudio")}>
          <p className="option-title">Upload Audio</p>
          <img
            src="https://img.icons8.com/color/96/upload.png"
            alt="Upload Icon"
            className="option-icon"
          />
        </div>

        <div className="option-box" onClick={() => navigate("/Microphone")}>
          <p className="option-title">
            Voice <br /> Microphone
          </p>
          <img
            src="https://img.icons8.com/ios-filled/100/2c6e7f/microphone.png"
            alt="Microphone Icon"
            className="option-icon"
          />
        </div>
      </div>
    </div>
  );
}

export default Welcome;
