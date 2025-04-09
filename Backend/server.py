from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from pydub import AudioSegment
import numpy as np
import tensorflow as tf
from io import BytesIO
import pyodbc
from datetime import datetime

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'flac', 'mp3', 'wav', 'webm'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# MSSQL database connection setup
conn = pyodbc.connect(
    'DRIVER={ODBC Driver 17 for SQL Server};'
    'SERVER=LAPTOP-H0CKMDVC\\SQLEXPRESS;'
    'DATABASE=EmotionDB;'
    'Trusted_Connection=yes;'
)
cursor = conn.cursor()

# Create the results table if it doesn't exist
cursor.execute('''
    IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='AudioResults' AND xtype='U')
    CREATE TABLE AudioResults (
        id INT PRIMARY KEY IDENTITY(1,1),
        filename NVARCHAR(255),
        emotion NVARCHAR(50),
        timestamp DATETIME
    )
''')
conn.commit()

# Load your trained model
# model = tf.keras.models.load_model('path_to_your_model.h5')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_audio(file_obj):
    try:
        sound = AudioSegment.from_file(file_obj)
        sound = sound.set_frame_rate(16000).set_channels(1)
        samples = np.array(sound.get_array_of_samples()).astype(np.float32)
        samples = samples / np.max(np.abs(samples))  # Normalize
        return samples
    except Exception as e:
        print("Preprocessing error:", e)
        return None

@app.route('/predict-audio', methods=['POST'])
def predict_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file part"}), 400

    file = request.files['audio']

    if file and allowed_file(file.filename):
        audio_file = BytesIO(file.read())
        features = preprocess_audio(audio_file)

        if features is None:
            return jsonify({"error": "Error processing audio"}), 400

        features = np.expand_dims(features, axis=0)

        # Perform prediction (replace with real model)
        # prediction = model.predict(features)
        # emotion = np.argmax(prediction, axis=1)
        # For testing/demo purposes only:
        import random
        emotion = [random.randint(0, 4)]

        emotion_map = {
            0: "neutral",
            1: "anger",
            2: "happiness",
            3: "sadness",
            4: "frustration"
        }
        predicted_emotion = emotion_map.get(emotion[0], "unknown")

        # Log to MSSQL
        timestamp = datetime.now()
        cursor.execute(
            "INSERT INTO AudioResults (filename, emotion, timestamp) VALUES (?, ?, ?)",
            file.filename, predicted_emotion, timestamp
        )
        conn.commit()

        return jsonify({"emotion": predicted_emotion})

    else:
        return jsonify({"error": "Invalid file type"}), 400

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)

