from flask import Flask, request, jsonify
from flask_cors import CORS  # To handle CORS
import os
from pydub import AudioSegment
import numpy as np
import tensorflow as tf
from io import BytesIO

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load your trained model here (assumed to be a TensorFlow model for this example)
model = tf.keras.models.load_model('path_to_your_model.h5')

# Define the upload folder and allowed extensions
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'flac', 'mp3', 'wav', 'webm'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Prediction route for audio emotion recognition
@app.route('/predict-audio', methods=['POST'])
def predict_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file part"}), 400

    file = request.files['audio']
    
    if file and allowed_file(file.filename):
        # Process the audio file and make the prediction
        audio_file = BytesIO(file.read())  # Read file into memory
        features = preprocess_audio(audio_file)

        if features is None:
            return jsonify({"error": "Error processing audio"}), 400

        # Pass the extracted features to the model (assuming it's a TensorFlow model here)
        features = np.expand_dims(features, axis=0)  # Add batch dimension
        prediction = model.predict(features)
        
        # Map model output to emotion (adapt this as per your model's output)
        emotion = np.argmax(prediction, axis=1)  # Assuming multi-class classification
        
        emotion_map = {
            0: "neutral",
            1: "anger",
            2: "happiness",
            3: "sadness",
            4: "frustration"
        }
        predicted_emotion = emotion_map.get(emotion[0], "unknown")

        return jsonify({"emotion": predicted_emotion})

    else:
        return jsonify({"error": "Invalid file type"}), 400

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
