import React from "react";
import { Routes, Route } from "react-router-dom";
import Welcome from "./Welcome.jsx";
import UploadAudio from "./UploadAudio.jsx";
import Microphone from "./Microphone.jsx";

function App() {
  return (
    <Routes>
      <Route path="/" element={<Welcome />} />
      <Route path="/UploadAudio" element={<UploadAudio />} />
      <Route path="/microphone" element={<Microphone />} />
    </Routes>
  );
}

export default App;
