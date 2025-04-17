// import React, { useState } from "react";
// import { FaCloudUploadAlt } from "react-icons/fa";
// import "./App.css";

// function UploadAudio() {
//   const [file, setFile] = useState(null);
//   const [result, setResult] = useState("");
//   const [error, setError] = useState("");

//   const handleFileChange = (e) => {
//     const selected = e.target.files[0];
//     if (selected && selected.type === "audio/flac") {
//       setFile(selected);
//       setError("");
//       setResult("");
//     } else {
//       setFile(null);
//       setError("Please upload only FLAC files");
//     }
//   };

//   const handleCheck = async () => {
//     if (!file) {
//       setError("No file selected!");
//       return;
//     }

//     const formData = new FormData();
//     formData.append("audio", file);

//     try {
//       const res = await fetch("http://localhost:5000/predict-audio", {
//         method: "POST",
//         body: formData,
//       });

//       const data = await res.json();

//       if (!res.ok) throw new Error(data.error || "Prediction failed");

//       // Ensure consistent emotion labels (Happy, Sad, Frustrated, Neutral, Angry)
//       setResult(data.emotion);
//       setError("");
//     } catch (err) {
//       setError(err.message);
//       console.error("API Error:", err);
//     }
//   };

//   // Emotion class mapping (must match backend labels exactly)
//   const getEmotionClass = (emotion) => {
//     const emotionMap = {
//       Happy: "emotion-happy",
//       Sad: "emotion-sad",
//       Frustrated: "emotion-frustrated",
//       Neutral: "emotion-neutral",
//       Angry: "emotion-angry",
//     };
//     return emotionMap[emotion] || "";
//   };

//   return (
//     <div className="container">
//       <div className="static-box">
//         <h2 className="option-title">Upload Audio</h2>

//         <div className="upload-container">
//           <label className="upload-box-large">
//             <FaCloudUploadAlt style={{ marginRight: 10 }} />
//             Upload Audio
//             <input
//               type="file"
//               accept=".flac"
//               onChange={handleFileChange}
//               hidden
//             />
//           </label>

//           <button className="check-btn" onClick={handleCheck}>
//             Analyze
//           </button>
//         </div>

//         {file && (
//           <div style={{ marginTop: 20 }}>
//             <p style={{ fontWeight: "bold" }}>Preview:</p>
//             <audio controls src={URL.createObjectURL(file)} />
//           </div>
//         )}

//         {error && <p className="error-message">{error}</p>}

//         <div className="result-section">
//           <h3>Result</h3>
//           {result && (
//             <>
//               <p>Detected emotion:</p>
//               <div className={`result-box ${getEmotionClass(result)}`}>
//                 {result}
//               </div>
//             </>
//           )}
//         </div>
//       </div>
//     </div>
//   );
// }

// export default UploadAudio;

import React, { useState } from "react";
import { FaCloudUploadAlt } from "react-icons/fa";
import "./App.css";

function UploadAudio() {
  const [file, setFile] = useState(null);
  const [emotion, setEmotion] = useState("");
  const [confidence, setConfidence] = useState(null);
  const [allPredictions, setAllPredictions] = useState({});
  const [error, setError] = useState("");

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (selected && selected.type === "audio/flac") {
      setFile(selected);
      setError("");
      setEmotion("");
      setConfidence(null);
      setAllPredictions({});
    } else {
      setFile(null);
      setError("Please upload only FLAC files");
    }
  };

  const handleCheck = async () => {
    if (!file) {
      setError("No file selected!");
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

      if (!res.ok) throw new Error(data.error || "Prediction failed");

      setEmotion(data.emotion);
      setConfidence(data.confidence);
      setAllPredictions(data.all_predictions);
      setError("");
    } catch (err) {
      setError(err.message);
      console.error("API Error:", err);
    }
  };

  const formatConfidence = (value) => (value * 100).toFixed(2) + "%";

  const getEmotionClass = (emotion) => {
    const emotionMap = {
      Happy: "emotion-happy",
      Sad: "emotion-sad",
      Frustrated: "emotion-frustrated",
      Neutral: "emotion-neutral",
      Angry: "emotion-angry",
    };
    return emotionMap[emotion] || "";
  };

  return (
    <div className="container">
      <div className="static-box">
        <h2 className="option-title">Upload Audio</h2>

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
            Analyze
          </button>
        </div>

        {file && (
          <div style={{ marginTop: 20 }}>
            <p style={{ fontWeight: "bold" }}>Preview:</p>
            <audio controls src={URL.createObjectURL(file)} />
          </div>
        )}

        {error && <p className="error-message">{error}</p>}

        <div className="result-section">
          {emotion && (
            <div className="results">
              <h3>Result</h3>

              <div className="results-summary">
                <p>
                  Dominant Emotion:{" "}
                  <span className={`emotion-text ${getEmotionClass(emotion)}`}>
                    {emotion}
                  </span>
                </p>
                <p>
                  Confidence:{" "}
                  <span className="confidence-value">
                    {formatConfidence(confidence)}
                  </span>
                </p>
              </div>

              <div className="predictions">
                <h3>All Predictions:</h3>
                <ul className="predictions-list">
                  {Object.entries(allPredictions).map(([emotion, value]) => (
                    <li key={emotion} className="prediction-item">
                      <span className="emotion-name">{emotion}:</span>
                      <span className="emotion-confidence">
                        {formatConfidence(value)}
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default UploadAudio;
