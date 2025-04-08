# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import os
# from pydub import AudioSegment
# import numpy as np
# import tensorflow as tf
# from io import BytesIO
# import pyodbc
# from datetime import datetime

# app = Flask(__name__)
# CORS(app)

# UPLOAD_FOLDER = './uploads'
# ALLOWED_EXTENSIONS = {'flac', 'mp3', 'wav', 'webm'}
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # MSSQL database connection setup
# conn = pyodbc.connect(
#     'DRIVER={ODBC Driver 17 for SQL Server};'
#     'SERVER=LAPTOP-H0CKMDVC\\SQLEXPRESS;'
#     'DATABASE=EmotionDB;'
#     'Trusted_Connection=yes;'
# )
# cursor = conn.cursor()

# # Create the results table if it doesn't exist
# cursor.execute('''
#     IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='AudioResults' AND xtype='U')
#     CREATE TABLE AudioResults (
#         id INT PRIMARY KEY IDENTITY(1,1),
#         filename NVARCHAR(255),
#         emotion NVARCHAR(50),
#         timestamp DATETIME
#     )
# ''')
# conn.commit()

# # Load your trained model
# # model = tf.keras.models.load_model('path_to_your_model.h5')

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# def preprocess_audio(file_obj):
#     try:
#         sound = AudioSegment.from_file(file_obj)
#         sound = sound.set_frame_rate(16000).set_channels(1)
#         samples = np.array(sound.get_array_of_samples()).astype(np.float32)
#         samples = samples / np.max(np.abs(samples))  # Normalize
#         return samples
#     except Exception as e:
#         print("Preprocessing error:", e)
#         return None

# @app.route('/predict-audio', methods=['POST'])
# def predict_audio():
#     if 'audio' not in request.files:
#         return jsonify({"error": "No audio file part"}), 400

#     file = request.files['audio']

#     if file and allowed_file(file.filename):
#         audio_file = BytesIO(file.read())
#         features = preprocess_audio(audio_file)

#         if features is None:
#             return jsonify({"error": "Error processing audio"}), 400

#         features = np.expand_dims(features, axis=0)

#         # Perform prediction (replace with real model)
#         # prediction = model.predict(features)
#         # emotion = np.argmax(prediction, axis=1)
#         # For testing/demo purposes only:
#         import random
#         emotion = [random.randint(0, 4)]

#         emotion_map = {
#             0: "neutral",
#             1: "anger",
#             2: "happiness",
#             3: "sadness",
#             4: "frustration"
#         }
#         predicted_emotion = emotion_map.get(emotion[0], "unknown")

#         # Log to MSSQL
#         timestamp = datetime.now()
#         cursor.execute(
#             "INSERT INTO AudioResults (filename, emotion, timestamp) VALUES (?, ?, ?)",
#             file.filename, predicted_emotion, timestamp
#         )
#         conn.commit()

#         return jsonify({"emotion": predicted_emotion})

#     else:
#         return jsonify({"error": "Invalid file type"}), 400

# if __name__ == '__main__':
#     if not os.path.exists(UPLOAD_FOLDER):
#         os.makedirs(UPLOAD_FOLDER)
#     app.run(debug=True)

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import librosa
import os
import pyodbc
from datetime import datetime

# Initialize Flask
app = Flask(__name__)
CORS(app)

# Database Setup
conn = pyodbc.connect(
    'DRIVER={ODBC Driver 17 for SQL Server};'
    'SERVER=LAPTOP-H0CKMDVC\\SQLEXPRESS;'
    'DATABASE=EmotionDB;'
    'Trusted_Connection=yes;'
)
cursor = conn.cursor()

# Load Models
model_3class = tf.keras.models.load_model('ResNet_BiLSTM_Attention_080_3classes.h5')
model_5class = tf.keras.models.load_model('ResNet_BiLSTM_Attention_0735_test2.h5')

# Emotion Mappings (MUST match model training labels exactly)
EMOTION_MAPPING = {
    '3class': ["Happy", "Neutral", "Negative"],
    '5class': ["Happy", "Sad", "Frustrated", "Neutral", "Angry"]
}

# Audio Preprocessing (Critical for both models)
def preprocess_audio(file_path):
    try:
        # Parameters (must match what models were trained on)
        TARGET_SR = 22050      # Sample rate
        FRAME_LENGTH = 2048    # Number of samples per frame
        HOP_LENGTH = 512       # Sliding window for frames
        N_MELS = 128           # Mel bands
        DURATION = 3           # Seconds (adjust if models expect different length)
        
        # Load and resample audio
        y, sr = librosa.load(file_path, sr=TARGET_SR, duration=DURATION)
        
        # Extract features (same for both models)
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=FRAME_LENGTH, 
            hop_length=HOP_LENGTH, n_mels=N_MELS
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        # MFCCs
        mfcc = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=13,
            n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH
        )
        
        # Zero-crossing rate
        zcr = librosa.feature.zero_crossing_rate(
            y, frame_length=FRAME_LENGTH, 
            hop_length=HOP_LENGTH
        )
        
        # Combine features
        features = np.vstack([
            zcr,
            mel_db,
            mfcc
        ])
        
        # Pad/trim to fixed size (models expect consistent input shape)
        max_frames = 130  # Adjust based on DURATION and HOP_LENGTH
        features = librosa.util.fix_length(features, size=max_frames, axis=1)
        
        return features.T  # Transpose to (timesteps, features)

    except Exception as e:
        print(f"Preprocessing failed: {str(e)}")
        return None

# API Routes
@app.route('/predict-3class', methods=['POST'])
def predict_3class():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file"}), 400
    
    file = request.files['audio']
    temp_path = "temp_audio.wav"
    file.save(temp_path)
    
    # Preprocess (required - models expect features, not raw audio)
    features = preprocess_audio(temp_path)
    os.remove(temp_path)
    
    if features is None:
        return jsonify({"error": "Audio processing failed"}), 400
    
    # Reshape for model (add batch dimension)
    features = np.expand_dims(features, axis=0)
    
    # Predict
    prediction = model_3class.predict(features)
    emotion_idx = np.argmax(prediction)
    predicted_emotion = EMOTION_MAPPING['3class'][emotion_idx]
    
    # Log to database
    cursor.execute(
        """INSERT INTO AudioResults 
        (filename, model_type, emotion, confidence, timestamp) 
        VALUES (?, ?, ?, ?, ?)""",
        (file.filename, "3class", predicted_emotion, 
         float(np.max(prediction)), datetime.now()
    ))
    conn.commit()
    
    return jsonify({
        "model": "3class",
        "emotion": predicted_emotion,
        "confidence": float(np.max(prediction))
    })

@app.route('/predict-5class', methods=['POST'])
def predict_5class():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file"}), 400
    
    file = request.files['audio']
    temp_path = "temp_audio.wav"
    file.save(temp_path)
    
    features = preprocess_audio(temp_path)
    os.remove(temp_path)
    
    if features is None:
        return jsonify({"error": "Audio processing failed"}), 400
    
    features = np.expand_dims(features, axis=0)
    prediction = model_5class.predict(features)
    emotion_idx = np.argmax(prediction)
    predicted_emotion = EMOTION_MAPPING['5class'][emotion_idx]
    
    cursor.execute(
        """INSERT INTO AudioResults 
        (filename, model_type, emotion, confidence, timestamp) 
        VALUES (?, ?, ?, ?, ?)""",
        (file.filename, "5class", predicted_emotion, 
         float(np.max(prediction)), datetime.now()
    ))
    conn.commit()
    
    return jsonify({
        "model": "5class",
        "emotion": predicted_emotion,
        "confidence": float(np.max(prediction))
    })

if __name__ == '__main__':
    # Verify model input shapes match preprocessing output
    print("3-class model input shape:", model_3class.input_shape)
    print("5-class model input shape:", model_5class.input_shape)
    app.run(host='0.0.0.0', port=5000, debug=True)