import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers, models
from tensorflow.keras.optimizers import Adam
import os
from tensorflow.keras import layers, regularizers, models
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import os
import numpy as np


def residual_block(x, filters, kernel_size, dilation_rate=1, dropout_rate=0.3, regularizer=regularizers.l2(0.001)):
    shortcut = x
    # Convolution 1 with dilated convolution
    x = layers.Conv2D(filters, kernel_size, padding='same', activation='relu',
                      dilation_rate=dilation_rate, kernel_regularizer=regularizer)(x)
    x = layers.BatchNormalization()(x)
    # Convolution 2 withoutactivation before BatchNormalization
    x = layers.Conv2D(filters, kernel_size, padding='same', activation=None,
                      dilation_rate=dilation_rate, kernel_regularizer=regularizer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([shortcut, x])
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    return x

class AttentionLayer(layers.Layer):
    def __init__(self, units=1):
        super(AttentionLayer, self).__init__()
        self.dense_score = layers.Dense(units, activation='tanh')
        self.dense_attention = layers.Dense(1)

    def call(self, inputs):
        score = self.dense_score(inputs)
        attention_weights = layers.Softmax(axis=1)(self.dense_attention(score))
        context_vector = layers.Multiply()([inputs, attention_weights])
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector


def process_audio(file):
    y, sr = librosa.load(file, sr=22050)
    frame_length = 2048
    hop_length = 512
    n_mels = 128
    max_frames = 3826  

    # calculate feature
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=frame_length, hop_length=hop_length)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)

    
    combined = np.vstack((zcr, mel_db, mfcc, spectral_centroid, spectral_rolloff, rms))
    combined = librosa.util.fix_length(combined, size=max_frames, axis=1)

    # Normalize
    min_vals = np.min(combined, axis=1, keepdims=True)
    max_vals = np.max(combined, axis=1, keepdims=True)
    combined = (combined - min_vals) / (max_vals - min_vals + 1e-8)

    # transform to 2D array
    feature = combined.T.astype(np.float32)
    return feature

def build_hybrid_model(metadata):
    input_shape = (metadata['max_frames'], metadata['n_features'], 1)
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
    
    # BiLSTM Layer
    x = layers.Bidirectional(layers.LSTM(512, return_sequences=True, kernel_regularizer=regularizers.l2(0.001)))(x)
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, kernel_regularizer=regularizers.l2(0.001)))(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, kernel_regularizer=regularizers.l2(0.001)))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, kernel_regularizer=regularizers.l2(0.001)))(x)
    
    # Attention Mechanism
    context_vector = AttentionLayer()(x)
    
    # Classifier
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(context_vector)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(len(metadata['classes']), activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model


@st.cache_resource
def load_trained_model():
  
    metadata = np.load('/Users/titiphonphunmongkon/Documents/Data_mining/project/metadata_normalized.npy', allow_pickle=True).item()

   
    model = build_hybrid_model(metadata)
    
    model.load_weights('/Users/titiphonphunmongkon/Documents/Data_mining/project/4layer_weight.h5')
    return model


def predict_audio(model, feature):
    feature = np.expand_dims(feature, axis=-1)
    pred = model.predict(np.array([feature]))  # ทำนาย
    class_idx = np.argmax(pred, axis=1)[0]
    return class_idx

# Streamlit UI
st.title("Audio Classification with Streamlit")

uploaded_file = st.file_uploader("Upload your .flac audio file", type=["flac"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    
    feature = process_audio(uploaded_file)
    st.write("Audio processed successfully!")

    model = load_trained_model()

    predicted_class = predict_audio(model, feature)

    metadata = np.load('metadata_normalized.npy', allow_pickle=True).item()
    predicted_label = metadata['classes'][predicted_class]
    
    st.write(f"Predicted Class: {predicted_label}")

