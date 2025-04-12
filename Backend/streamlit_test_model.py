import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers, models
import os

# Static metadata - defined once at the top
METADATA = {
    'max_frames': 3826,
    'n_features': 145,
    'classes': ['Happy', 'Sad', 'Frustrated', 'Neutral', 'Angry'],
    'sr': 22050,
    'frame_length': 2048,
    'hop_length': 512,
    'n_mels': 128
}

def residual_block(x, filters, kernel_size, dilation_rate=1, dropout_rate=0.3, regularizer=regularizers.l2(0.001)):
    shortcut = x
    x = layers.Conv2D(filters, kernel_size, padding='same', activation='relu',
                      dilation_rate=dilation_rate, kernel_regularizer=regularizer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, kernel_size, padding='same', activation=None,
                      dilation_rate=dilation_rate, kernel_regularizer=regularizer)(x)
    x = layers.BatchNormalization()(x)
    
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1,1), padding='same')(shortcut)
    
    x = layers.Add()([shortcut, x])
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    return x

class AttentionLayer(layers.Layer):
    def __init__(self):
        super(AttentionLayer, self).__init__()
        self.dense_score = layers.Dense(1, activation='tanh')
        self.dense_attention = layers.Dense(1)

    def call(self, inputs):
        score = self.dense_score(inputs)
        attention_weights = tf.nn.softmax(self.dense_attention(score), axis=1)
        context_vector = tf.reduce_sum(inputs * attention_weights, axis=1)
        return context_vector

def process_audio(file):
    y, sr = librosa.load(file, sr=METADATA['sr'])
    features = []
    
    # Extract all features
    features.append(librosa.feature.zero_crossing_rate(
        y, frame_length=METADATA['frame_length'], hop_length=METADATA['hop_length']))
    
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=METADATA['frame_length'], 
        hop_length=METADATA['hop_length'], n_mels=METADATA['n_mels'])
    features.append(librosa.power_to_db(mel, ref=np.max))
    
    features.append(librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=13, 
        n_fft=METADATA['frame_length'], hop_length=METADATA['hop_length']))
    
    features.append(librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=METADATA['frame_length'], hop_length=METADATA['hop_length']))
    
    features.append(librosa.feature.spectral_rolloff(
        y=y, sr=sr, n_fft=METADATA['frame_length'], hop_length=METADATA['hop_length']))
    
    features.append(librosa.feature.rms(
        y=y, frame_length=METADATA['frame_length'], hop_length=METADATA['hop_length']))
    
    # Combine and fix length
    combined = np.vstack(features)
    combined = librosa.util.fix_length(combined, size=METADATA['max_frames'], axis=1)
    
    # Normalize
    combined = (combined - combined.min(axis=1, keepdims=True)) / \
               (combined.max(axis=1, keepdims=True) - combined.min(axis=1, keepdims=True) + 1e-8)
    
    return combined.T.astype(np.float32)

def build_hybrid_model():
    input_shape = (METADATA['max_frames'], METADATA['n_features'], 1)
    inputs = layers.Input(shape=input_shape)
    
    # CNN Part
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(0.001))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.4)(x)
    
    # Residual block
    x = residual_block(x, filters=64, kernel_size=(3,3), dilation_rate=2, dropout_rate=0.4)
    
    # CNN Part 2
    x = layers.Conv2D(128, (3,3), activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.5)(x)
    
    # TimeDistributed Flatten
    x = layers.TimeDistributed(layers.Flatten())(x)
    
    # BiLSTM Layers
    x = layers.Bidirectional(layers.LSTM(512, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    
    # Attention Mechanism
    context_vector = AttentionLayer()(x)
    
    # Classifier
    x = layers.Dense(256, activation='relu')(context_vector)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(len(METADATA['classes']), activation='softmax')(x)
    
    return models.Model(inputs=inputs, outputs=outputs)

@st.cache_resource
def load_trained_model():
    model = build_hybrid_model()
    model.load_weights(r"C:\Users\cmanw\Downloads\4layer_weight.h5")
    return model

def predict_emotion(model, audio_file):
    features = process_audio(audio_file)
    features = np.expand_dims(features, axis=(0, -1))  # Add batch and channel dims
    predictions = model.predict(features)[0]  # Get the first (and only) prediction
    
    # Create a dictionary of emotion-confidence pairs
    emotion_confidences = {
        METADATA['classes'][i]: float(confidence) 
        for i, confidence in enumerate(predictions)
    }
    
    return emotion_confidences

# Modified Streamlit UI section
st.title("Speech Emotion Recognition")

uploaded_file = st.file_uploader("Upload FLAC audio file", type=["flac"])
if uploaded_file:
    st.audio(uploaded_file)
    
    if st.button("Analyze Emotion"):
        model = load_trained_model()
        confidences = predict_emotion(model, uploaded_file)
        
        st.subheader("Emotion Analysis Results")
        
        # Display dominant emotion
        dominant_emotion = max(confidences.items(), key=lambda x: x[1])
        st.metric("Dominant Emotion", 
                 dominant_emotion[0], 
                 f"{dominant_emotion[1]*100:.1f}% confidence")
        
        # Display confidence bars for all emotions
        st.subheader("Confidence Levels")
        for emotion, confidence in confidences.items():
            st.write(f"{emotion}")
            st.progress(confidence)
            st.write(f"{confidence*100:.1f}%")
            st.write("---")