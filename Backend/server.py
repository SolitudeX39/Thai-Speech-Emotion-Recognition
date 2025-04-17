# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import os
# from datetime import datetime
# import pyodbc
# from model_handler import ThaiEmotionRecognizer
# from io import BytesIO
# import subprocess

# app = Flask(__name__)

# # CORS Configuration
# # CORS(app, resources={
# #     r"/predict-audio": {
# #         "origins": ["http://localhost:5174"],
# #         "methods": ["POST"],
# #         "allow_headers": ["Content-Type"]
# #     }
# # })
# # CORS(app, resources={
# #     r"/predict-wav": {
# #         "origins": ["http://localhost:5174"],
# #         "methods": ["POST"],
# #         "allow_headers": ["Content-Type"],
# #         "expose_headers": ["Content-Type"]
# #     }
# # })
# # แทนที่ CORS เดิมทั้งหมดด้วยบรรทัดนี้
# CORS(app, resources={r"/*": {"origins": "*"}})

# @app.after_request
# def after_request(response):
#     response.headers.add('Access-Control-Allow-Credentials', 'true')
#     return response

# # Model Initialization
# model = ThaiEmotionRecognizer()

# # Database configuration
# UPLOAD_FOLDER = './uploads'
# ALLOWED_EXTENSIONS = {'flac', 'mp3', 'wav', 'webm'}
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# def get_db_connection():
#     try:
#         conn = pyodbc.connect(
#             'DRIVER={ODBC Driver 17 for SQL Server};'
#             'SERVER=LAPTOP-H0CKMDVC\\SQLEXPRESS;'
#             'DATABASE=EmotionDB;'
#             'Trusted_Connection=yes;'
#         )
#         print("✅ DB connected")
#         return conn
#     except Exception as e:
#         print("❌ DB connection error:", e)
#         raise


# def init_db():
#     with get_db_connection() as conn:
#         cursor = conn.cursor()
#         cursor.execute('''
#             IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='AudioResults' AND xtype='U')
#             CREATE TABLE AudioResults (
#                 id INT PRIMARY KEY IDENTITY(1,1),
#                 filename NVARCHAR(255),
#                 emotion NVARCHAR(50),
#                 confidence FLOAT,
#                 timestamp DATETIME
#             )
#         ''')
#         conn.commit()

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# @app.route('/predict-audio', methods=['POST'])
# def predict_audio():
#     if 'audio' not in request.files:
#         return jsonify({"error": "No audio file provided"}), 400

#     file = request.files['audio']
#     if not file or not allowed_file(file.filename):
#         return jsonify({"error": "Invalid file type"}), 400

#     try:
#         # Use the model handler for prediction
#         audio_bytes = file.read()
#         predictions = model.predict(audio_bytes)
        
#         # Get dominant emotion
#         dominant_emotion, confidence = max(predictions.items(), key=lambda x: x[1])
        
#         # Log to database
#         with get_db_connection() as conn:
#             cursor = conn.cursor()
#             cursor.execute(
#                 "INSERT INTO AudioResults (filename, emotion, confidence, timestamp) VALUES (?, ?, ?, ?)",
#                 file.filename, dominant_emotion, float(confidence), datetime.now()
#             )
#             conn.commit()

#         return jsonify({
#             "status": "success",
#             "emotion": dominant_emotion,
#             "confidence": confidence,
#             "all_predictions": predictions
#         })

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
    
# @app.route('/predict-wav', methods=['POST'])
# def predict_wav():
#     if 'audio' not in request.files:
#         return jsonify({"error": "No audio file"}), 400

#     file = request.files['audio']
#     if not file.filename.lower().endswith('.wav'):
#         return jsonify({"error": "Only WAV files accepted"}), 400

#     try:
#         # Directly process WAV without conversion
#         audio_bytes = file.read()
#         predictions = model.predict(audio_bytes)
#         dominant_emotion = max(predictions.items(), key=lambda x: x[1])
        
#         return jsonify({
#             "status": "success",
#             "emotion": dominant_emotion[0],
#             "confidence": dominant_emotion[1],
#             "all_predictions": predictions
#         })
        
#     except Exception as e:
#         app.logger.error(f"Prediction error: {str(e)}")
#         return jsonify({"error": str(e)}), 500


# if __name__ == '__main__':
#     # Initialize
#     os.makedirs(UPLOAD_FOLDER, exist_ok=True)
#     init_db()
    
#     # Run server
#     app.run(host='0.0.0.0', port=5000, debug=True)

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from datetime import datetime
import pyodbc
from model_handler import ThaiEmotionRecognizer

app = Flask(__name__)

# CORS: อนุญาตทุก origin
CORS(app, resources={r"/*": {"origins": "*"}})

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

# Initialize Model
model = ThaiEmotionRecognizer()

# Database settings
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'flac', 'mp3', 'wav', 'webm'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def get_db_connection():
    try:
        conn = pyodbc.connect(
            'DRIVER={ODBC Driver 17 for SQL Server};'
            'SERVER=LAPTOP-H0CKMDVC\\SQLEXPRESS;'
            'DATABASE=EmotionDB;'
            'Trusted_Connection=yes;'
        )
        print("✅ DB connected")
        return conn
    except Exception as e:
        print("❌ DB connection error:", e)
        raise

def init_db():
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='AudioResults' AND xtype='U')
            CREATE TABLE AudioResults (
                id INT PRIMARY KEY IDENTITY(1,1),
                filename NVARCHAR(255),
                emotion NVARCHAR(50),
                confidence FLOAT,
                timestamp DATETIME
            )
        ''')
        conn.commit()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict-audio', methods=['POST'])
def predict_audio():
    print("🎯 /predict-audio called")

    if 'audio' not in request.files:
        print("❌ No audio part in request")
        return jsonify({"error": "No audio file provided"}), 400

    file = request.files['audio']
    print(f"📂 Received file: {file.filename}")

    if not file or not allowed_file(file.filename):
        print("❌ Invalid file or unsupported type:", file.filename)
        return jsonify({"error": "Invalid file type"}), 400

    try:
        audio_bytes = file.read()

        # Run model prediction
        predictions = model.predict(audio_bytes)
        print("📊 Predictions:", predictions)

        if not predictions:
            print("❌ No predictions returned from model.")
            return jsonify({"error": "Model did not return predictions"}), 500

        dominant_emotion, confidence = max(predictions.items(), key=lambda x: x[1])
        print(f"🏆 Dominant: {dominant_emotion} ({confidence:.2f})")

        # Save to DB
        with get_db_connection() as conn:
            cursor = conn.cursor()
            print("📝 Inserting into DB:", file.filename, dominant_emotion, confidence)

            cursor.execute(
                "INSERT INTO AudioResults (filename, emotion, confidence, timestamp) VALUES (?, ?, ?, ?)",
                file.filename, dominant_emotion, float(confidence), datetime.now()
            )
            conn.commit()
            print("✅ Inserted to DB")

        return jsonify({
            "status": "success",
            "emotion": dominant_emotion,
            "confidence": confidence,
            "all_predictions": predictions
        })

    except Exception as e:
        print("❌ Exception during prediction or DB:", str(e))
        return jsonify({"error": str(e)}), 500

@app.route('/predict-wav', methods=['POST'])
def predict_wav():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file"}), 400

    file = request.files['audio']
    if not file.filename.lower().endswith('.wav'):
        return jsonify({"error": "Only WAV files accepted"}), 400

    try:
        audio_bytes = file.read()
        predictions = model.predict(audio_bytes)
        print("📊 Predictions:", predictions)

        if not predictions:
            return jsonify({"error": "Model did not return predictions"}), 500

        dominant_emotion, confidence = max(predictions.items(), key=lambda x: x[1])
        print(f"🏆 Dominant: {dominant_emotion} ({confidence:.2f})")

        # ✅ Save to DB
        with get_db_connection() as conn:
            cursor = conn.cursor()
            print("📝 Inserting into DB:", file.filename, dominant_emotion, confidence)

            cursor.execute(
                "INSERT INTO AudioResults (filename, emotion, confidence, timestamp) VALUES (?, ?, ?, ?)",
                file.filename, dominant_emotion, confidence, datetime.now()
            )

            conn.commit()
            print("✅ Inserted to DB")

        return jsonify({
            "status": "success",
            "emotion": dominant_emotion,
            "confidence": confidence,
            "all_predictions": predictions
        })

    except Exception as e:
        print("❌ Exception during /predict-wav:", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    init_db()
    app.run(host='0.0.0.0', port=5000, debug=True)
