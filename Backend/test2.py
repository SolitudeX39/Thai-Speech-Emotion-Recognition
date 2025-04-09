import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
from tensorflow.keras import layers, models, regularizers

# Load the model
def load_model():
    model = build_hybrid_model(metadata)  # Assuming build_hybrid_model function is already defined
    model.load_weights(r"C:\Users\cmanw\Downloads\ResNet_BiLSTM_Attention_0735_test2.h5")
    return model

# Load the metadata
def load_metadata():
    metadata = np.load("C:/Users/cmanw/OneDrive/Documents/Emotion-Recognition/Thai-Speech-Emotion-Recognition/Backend/metadata_normalized.npy", allow_pickle=True).item()
    return metadata

# Residual block (dilated convolution) function
def residual_block(x, filters, kernel_size, dilation_rate=1, dropout_rate=0.3, regularizer=regularizers.l2(0.001)):
    shortcut = x
    x = layers.Conv2D(filters, kernel_size, padding='same', activation='relu',
                      dilation_rate=dilation_rate, kernel_regularizer=regularizer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, kernel_size, padding='same', activation=None,
                      dilation_rate=dilation_rate, kernel_regularizer=regularizer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([shortcut, x])
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    return x

# Attention Block Function
def attention_block(inputs):
    score = layers.Dense(inputs.shape[-1], activation='tanh')(inputs)
    score = layers.Dense(1)(score)
    attention_weights = layers.Softmax(axis=1)(score)
    context_vector = layers.Multiply()([inputs, attention_weights])
    context_vector = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1), output_shape=(inputs.shape[0], inputs.shape[2]))(context_vector)
    return context_vector

# Build the hybrid model with the attention block and residual layers
def build_hybrid_model(metadata):
    input_shape = (metadata['max_frames'], metadata['n_features'], 1)
    inputs = layers.Input(shape=input_shape)
    
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(0.001))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.4)(x)
    
    x = residual_block(x, filters=64, kernel_size=(3,3), dilation_rate=2, dropout_rate=0.4)
    
    x = layers.Conv2D(128, (3,3), activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.TimeDistributed(layers.Flatten())(x)
    
    # Use Bidirectional LSTM (biLSTM)
    x = layers.Bidirectional(
            layers.LSTM(256, return_sequences=True, kernel_regularizer=regularizers.l2(0.001))
        )(x)
    x = layers.Bidirectional(
            layers.LSTM(128, return_sequences=True, kernel_regularizer=regularizers.l2(0.001))
        )(x)
    x = layers.Bidirectional(
            layers.LSTM(64, return_sequences=True, kernel_regularizer=regularizers.l2(0.001))
        )(x)
    
    # Apply Attention Block
    context_vector = attention_block(x)
    
    # Classifier
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(context_vector)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(len(metadata['classes']), activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

# Streamlit UI
st.title("Emotion Recognition from Audio")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "flac"])

if uploaded_file is not None:
    # Load model and metadata
    metadata = load_metadata()
    model = load_model()

    # Load and preprocess audio file
    audio_data, sr = librosa.load(uploaded_file, sr=22050)

    # Extract features like MFCC, Mel-spectrogram, etc.
    frame_length = 2048
    hop_length = 512
    n_mels = 128

    # Mel Spectrogram
    mel = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_fft=frame_length, hop_length=hop_length, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Other Features
    zcr = librosa.feature.zero_crossing_rate(y=audio_data, frame_length=frame_length, hop_length=hop_length)
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13, n_fft=frame_length, hop_length=hop_length)
    spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr, n_fft=frame_length, hop_length=hop_length)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr, n_fft=frame_length, hop_length=hop_length)
    rms = librosa.feature.rms(y=audio_data, frame_length=frame_length, hop_length=hop_length)

    # Stack features to get a total of 145 features (128 Mel + 17 additional features)
    features = np.vstack([mel_db, zcr, mfcc, spectral_centroid, spectral_rolloff, rms])
    features = features.T  # Transpose to (time_steps, features)
    
    # Padding to max_frames if necessary
    max_frames = metadata['max_frames']
    if features.shape[0] < max_frames:
        padding = np.zeros((max_frames - features.shape[0], features.shape[1]))
        features = np.vstack([features, padding])  # Pad the sequence to match max_frames

    # Ensure correct feature dimension (n_features, max_frames)
    features = np.expand_dims(features, axis=-1)  # Add channel dimension for model input
    features = np.expand_dims(features, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(features)

    # Display results
    predicted_class = metadata['classes'][np.argmax(prediction)]
    confidence = np.max(prediction)

    st.write(f"Predicted Emotion: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}")
