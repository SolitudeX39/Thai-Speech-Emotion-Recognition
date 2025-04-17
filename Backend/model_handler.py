import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, regularizers, models
from io import BytesIO
import os

class ThaiEmotionRecognizer:
    def __init__(self):
        self.METADATA = {
            'max_frames': 3826,
            'n_features': 145,
            'classes': ['Happy', 'Sad', 'Frustrated', 'Neutral', 'Angry'],
            'sr': 22050,
            'frame_length': 2048,
            'hop_length': 512,
            'n_mels': 128
        }
        
        # Build model and load weights
        self.model = self._build_hybrid_model()
        weights_path = os.path.join(os.path.dirname(__file__), r"C:\Users\Janejojija\Downloads\4layer_weight.h5")
        self.model.load_weights(weights_path)

    # 1. Residual Block (EXACTLY as in your code)
    def _residual_block(self, x, filters, kernel_size, dilation_rate=1, dropout_rate=0.3):
        shortcut = x
        x = layers.Conv2D(filters, kernel_size, padding='same', activation='relu',
                         dilation_rate=dilation_rate, kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, kernel_size, padding='same', activation=None,
                         dilation_rate=dilation_rate, kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        
        if shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, (1,1), padding='same')(shortcut)
        
        x = layers.Add()([shortcut, x])
        x = layers.Activation('relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        return x

    # 2. Attention Layer (EXACT implementation)
    class AttentionLayer(layers.Layer):
        def __init__(self):
            super().__init__()
            self.dense_score = layers.Dense(1, activation='tanh')
            self.dense_attention = layers.Dense(1)

        def call(self, inputs):
            score = self.dense_score(inputs)
            attention_weights = tf.nn.softmax(self.dense_attention(score), axis=1)
            context_vector = tf.reduce_sum(inputs * attention_weights, axis=1)
            return context_vector

    # 3. Model Architecture (EXACTLY as in your code)
    def _build_hybrid_model(self):
        input_shape = (self.METADATA['max_frames'], self.METADATA['n_features'], 1)
        inputs = layers.Input(shape=input_shape)
        
        # CNN Part
        x = layers.Conv2D(64, (3,3), activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(0.001))(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2,2))(x)
        x = layers.Dropout(0.4)(x)
        
        # Residual block
        x = self._residual_block(x, 64, (3,3), dilation_rate=2, dropout_rate=0.4)
        
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
        context_vector = self.AttentionLayer()(x)
        
        # Classifier
        x = layers.Dense(256, activation='relu')(context_vector)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(len(self.METADATA['classes']), activation='softmax')(x)
        
        return models.Model(inputs=inputs, outputs=outputs)

    # 4. EXACT Audio Processing (from your code)
    def process_audio(self, file_bytes):
        y, sr = librosa.load(BytesIO(file_bytes), sr=self.METADATA['sr'])
        features = []
        
        # Extract all features (EXACTLY as in your code)
        features.append(librosa.feature.zero_crossing_rate(
            y, frame_length=self.METADATA['frame_length'], 
            hop_length=self.METADATA['hop_length']))
        
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=self.METADATA['frame_length'], 
            hop_length=self.METADATA['hop_length'], n_mels=self.METADATA['n_mels'])
        features.append(librosa.power_to_db(mel, ref=np.max))
        
        features.append(librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=13, 
            n_fft=self.METADATA['frame_length'], hop_length=self.METADATA['hop_length']))
        
        features.append(librosa.feature.spectral_centroid(
            y=y, sr=sr, n_fft=self.METADATA['frame_length'], hop_length=self.METADATA['hop_length']))
        
        features.append(librosa.feature.spectral_rolloff(
            y=y, sr=sr, n_fft=self.METADATA['frame_length'], hop_length=self.METADATA['hop_length']))
        
        features.append(librosa.feature.rms(
            y=y, frame_length=self.METADATA['frame_length'], hop_length=self.METADATA['hop_length']))
        
        # Combine and fix length
        combined = np.vstack(features)
        combined = librosa.util.fix_length(combined, size=self.METADATA['max_frames'], axis=1)
        
        # Normalize (EXACT normalization from your code)
        combined = (combined - combined.min(axis=1, keepdims=True)) / \
                  (combined.max(axis=1, keepdims=True) - combined.min(axis=1, keepdims=True) + 1e-8)
        
        return combined.T.astype(np.float32)

    def predict(self, audio_bytes):
        features = self.process_audio(audio_bytes)
        features = np.expand_dims(features, axis=(0, -1))  # Add batch and channel dims
        predictions = self.model.predict(features)[0]
        return {
            emotion: float(confidence) 
            for emotion, confidence in zip(self.METADATA['classes'], predictions)
        }