import React, { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import './Microphone_UI.css';

const Microphone = () => {
  // State variables
  const [isRecording, setIsRecording] = useState(false);
  const [emotion, setEmotion] = useState(null);
  const [confidence, setConfidence] = useState(null);
  const [allPredictions, setAllPredictions] = useState(null);
  const [error, setError] = useState(null);
  const [audioURL, setAudioURL] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  
  // Refs
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const audioRef = useRef(null);
  const navigate = useNavigate();

  // Clean up on unmount
  useEffect(() => {
    return () => {
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
        mediaRecorderRef.current.stop();
      }
    };
  }, []);

  const startRecording = async () => {
    try {
      setError(null);
      setEmotion(null);
      setAudioURL('');
      audioChunksRef.current = [];
      
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      
      // Use default MediaRecorder (will typically use browser's default codec)
      mediaRecorderRef.current = new MediaRecorder(stream);
      
      mediaRecorderRef.current.ondataavailable = (event) => {
        audioChunksRef.current.push(event.data);
      };
      
      mediaRecorderRef.current.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current);
        
        // Convert to WAV in the browser using audiobuffer-to-wav
        const wavBlob = await convertToWav(audioBlob);
        const audioUrl = URL.createObjectURL(wavBlob);
        
        setAudioURL(audioUrl);
        sendAudioToAPI(wavBlob);
        stream.getTracks().forEach(track => track.stop());
      };
      
      mediaRecorderRef.current.start();
      setIsRecording(true);
      
      setTimeout(() => {
        if (mediaRecorderRef.current?.state === 'recording') {
          stopRecording();
        }
      }, 10000);
      
    } catch (err) {
      setError('Error accessing microphone: ' + err.message);
      console.error(err);
    }
  };
  
  // Add this helper function
  const convertToWav = async (blob) => {
    try {
      // Implementation using Web Audio API
      const arrayBuffer = await blob.arrayBuffer();
      const audioContext = new (window.AudioContext || window.webkitAudioContext)();
      const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
      
      // Convert to WAV
      const wavBlob = bufferToWav(audioBuffer);
      return wavBlob;
      
    } catch (err) {
      console.error("Conversion error:", err);
      return blob; // Fallback to original if conversion fails
    }
  };
  
  // Add this utility function
  const bufferToWav = (buffer) => {
    const numChannels = buffer.numberOfChannels;
    const sampleRate = buffer.sampleRate;
    const length = buffer.length;
    
    const interleaved = new Float32Array(length * numChannels);
    for (let channel = 0; channel < numChannels; channel++) {
      const channelData = buffer.getChannelData(channel);
      for (let i = 0; i < length; i++) {
        interleaved[i * numChannels + channel] = channelData[i];
      }
    }
    
    const wavBuffer = new ArrayBuffer(44 + interleaved.length * 2);
    const view = new DataView(wavBuffer);
    
    // Write WAV header
    const writeString = (view, offset, string) => {
      for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
      }
    };
    
    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + interleaved.length * 2, true);
    writeString(view, 8, 'WAVE');
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * numChannels * 2, true);
    view.setUint16(32, numChannels * 2, true);
    view.setUint16(34, 16, true);
    writeString(view, 36, 'data');
    view.setUint32(40, interleaved.length * 2, true);
    
    // Write PCM samples
    const volume = 1;
    let index = 44;
    for (let i = 0; i < interleaved.length; i++) {
      view.setInt16(index, interleaved[i] * (0x7FFF * volume), true);
      index += 2;
    }
    
    return new Blob([view], { type: 'audio/wav' });
  };
  

  // Stop recording function
  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const sendAudioToAPI = async (audioBlob) => {
    setIsLoading(true);
    setError(null);
    
    try {
      const formData = new FormData();
      formData.append('audio', audioBlob, 'recording.wav');
  
      // Add timeout and better error handling
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 30000); // 30s timeout
  
      const response = await fetch('http://localhost:5000/predict-wav', {
        method: 'POST',
        body: formData,
        signal: controller.signal,
        headers: {
          'Accept': 'application/json' // Ensure we expect JSON back
        }
      });
  
      clearTimeout(timeoutId);
  
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(
          errorData.error || 
          `Server error: ${response.status} ${response.statusText}`
        );
      }
  
      const data = await response.json();
      
      // Verify response structure
      if (!data.emotion || !data.confidence) {
        throw new Error("Invalid response format from server");
      }
  
      setEmotion(data.emotion);
      setConfidence(data.confidence);
      setAllPredictions(data.all_predictions || { [data.emotion]: data.confidence });
  
    } catch (err) {
      console.error('API Error:', err);
      setError(
        err.name === 'AbortError' 
          ? 'Request timed out (30s)' 
          : `Analysis failed: ${err.message}`
      );
    } finally {
      setIsLoading(false);
    }
  };

  // Format confidence percentage
  const formatConfidence = (value) => {
    return (value * 100).toFixed(1) + '%';
  };

  return (
    <div className="microphone-container">
      <div className="recording-card">
        <h1 className="title">Voice Emotion Analysis</h1>
        
        <div className="controls">
          <button
            onClick={isRecording ? stopRecording : startRecording}
            className={`record-button ${isRecording ? 'recording' : ''}`}
            disabled={isLoading}
          >
            {isRecording ? 'Stop Recording' : 'Start Recording'}
            {isLoading && <span className="spinner"></span>}
          </button>
          
          {isRecording && (
            <p className="recording-status">
              <span className="pulse-dot"></span>
              Recording... (auto-stops after 10 seconds)
            </p>
          )}
          
          {audioURL && (
            <div className="audio-player-container">
              <h2>Your Recording:</h2>
              <audio 
                ref={audioRef} 
                src={audioURL} 
                controls 
                className="audio-player"
              />
            </div>
          )}
          
          {isLoading && (
            <div className="loading-indicator">
              <div className="loader"></div>
              <p>Analyzing emotions...</p>
            </div>
          )}
          
          {error && (
            <div className="error-message">
              <p>Error: {error}</p>
            </div>
          )}
          
          {emotion && (
            <div className="results">
              <h2>Results</h2>
              <div className="results-summary">
                <p>
                  Dominant Emotion: <span className="emotion-text">{emotion}</span>
                </p>
                <p>
                  Confidence: <span className="confidence-value">{formatConfidence(confidence)}</span>
                </p>
              </div>
              
              <div className="predictions">
                <h3>All Predictions:</h3>
                <ul className="predictions-list">
                  {Object.entries(allPredictions).map(([emotion, value]) => (
                    <li key={emotion} className="prediction-item">
                      <span className="emotion-name">{emotion}:</span>
                      <span className="emotion-confidence">{formatConfidence(value)}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          )}
        </div>
        
        <button
          onClick={() => navigate('/UploadAudio')}
          className="back-button"
        >
          ← Try file upload instead
        </button>
      </div>
    </div>
  );
};

export default Microphone;