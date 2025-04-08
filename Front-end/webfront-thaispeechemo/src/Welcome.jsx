// import React from "react";
// import { useNavigate } from "react-router-dom";
// import "./App.css";

// function Welcome() {
//   const navigate = useNavigate();

//   return (
//     <div className="container">
//       <h1 className="title">
//         WELCOME to
//         <br />
//         Thai Speech Emotion Recognition
//       </h1>

//       <div className="button-container">
//         <div className="option-box" onClick={() => navigate("/UploadAudio")}>
//           <p className="option-title">Upload Audio</p>
//           <img
//             src="https://img.icons8.com/color/96/upload.png"
//             alt="Upload Icon"
//             className="option-icon"
//           />
//         </div>

//         <div className="option-box" onClick={() => navigate("/Microphone")}>
//           <p className="option-title">
//             Voice <br /> Microphone
//           </p>
//           <img
//             src="https://img.icons8.com/ios-filled/100/2c6e7f/microphone.png"
//             alt="Microphone Icon"
//             className="option-icon"
//           />
//         </div>
//       </div>
//     </div>
//   );
// }

// export default Welcome;

// import React, { useState } from "react";
// import { useNavigate } from "react-router-dom";
// import "./App.css";

// function Welcome() {
//   const [predictionModel, setPredictionModel] = useState("3class");
//   const navigate = useNavigate();

//   // Handle model selection
//   const handleModelChange = (e) => {
//     setPredictionModel(e.target.value);
//   };

//   // Navigate to selected model page (upload or microphone)
//   const handleNavigate = () => {
//     if (predictionModel === "3class") {
//       navigate("/UploadAudio"); // Simple prediction with 3 emotions
//     } else {
//       navigate("/Microphone"); // Complex prediction with 5 emotions
//     }
//   };

//   return (
//     <div className="container">
//       <h1 className="title">
//         WELCOME to
//         <br />
//         Thai Speech Emotion Recognition
//       </h1>

//       <div className="dropdown-container">
//         <select
//           value={predictionModel}
//           onChange={handleModelChange}
//           className="dropdown"
//         >
//           <option value="3class">Simplistic Prediction (3 emotions)</option>
//           <option value="5class">Complex Prediction (5 emotions)</option>
//         </select>
//       </div>

//       <div className="button-container">
//         <div className="option-box" onClick={handleNavigate}>
//           <p className="option-title">Upload Audio</p>
//           <img
//             src="https://img.icons8.com/color/96/upload.png"
//             alt="Upload Icon"
//             className="option-icon"
//           />
//         </div>

//         <div className="option-box" onClick={handleNavigate}>
//           <p className="option-title">
//             Voice <br /> Microphone
//           </p>
//           <img
//             src="https://img.icons8.com/ios-filled/100/2c6e7f/microphone.png"
//             alt="Microphone Icon"
//             className="option-icon"
//           />
//         </div>
//       </div>
//     </div>
//   );
// }

import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import "./App.css";

function Welcome() {
  const [predictionModel, setPredictionModel] = useState("");
  const navigate = useNavigate();

  // Handle model selection
  const handleModelChange = (e) => {
    setPredictionModel(e.target.value);
  };

  // Navigate to selected model page (upload or microphone)
  const handleNavigate = (destination) => {
    if (!predictionModel) {
      alert("Please select a prediction model before proceeding.");
      return;
    }

    // Navigate to the appropriate page based on the button clicked
    if (destination === "UploadAudio") {
      navigate("/UploadAudio"); // Always go to Upload Audio page for both models
    } else if (destination === "Microphone") {
      navigate("/Microphone"); // Always go to Microphone page for both models
    }
  };

  return (
    <div className="container">
      <h1 className="title">
        WELCOME to
        <br />
        Thai Speech Emotion Recognition
      </h1>

      <div className="dropdown-container">
        <select
          value={predictionModel}
          onChange={handleModelChange}
          className="dropdown"
        >
          <option value="" disabled>
            Select Prediction
          </option>
          <option value="3class">Simplistic Prediction (3 emotions)</option>
          <option value="5class">Complex Prediction (5 emotions)</option>
        </select>
      </div>

      <div className="button-container">
        <div
          className="option-box"
          onClick={() => handleNavigate("UploadAudio")}
        >
          <p className="option-title">Upload Audio</p>
          <img
            src="https://img.icons8.com/color/96/upload.png"
            alt="Upload Icon"
            className="option-icon"
          />
        </div>

        <div
          className="option-box"
          onClick={() => handleNavigate("Microphone")}
        >
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
