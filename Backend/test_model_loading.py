import os
import sys
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import librosa
from pathlib import Path
from collections import defaultdict

# Configure system encoding and reproducibility
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Set all random seeds
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# ======================
# 1. Model Components
# ======================

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

class FixedAttentionBlock(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def build(self, input_shape):
        self.dense1 = layers.Dense(input_shape[-1], activation='tanh')
        self.dense2 = layers.Dense(1)
        self.softmax = layers.Softmax(axis=1)
        self.multiply = layers.Multiply()
        super().build(input_shape)
        
    def call(self, inputs):
        attention = self.dense1(inputs)
        attention = self.dense2(attention)
        attention_weights = self.softmax(attention)
        context = self.multiply([inputs, attention_weights])
        return tf.reduce_sum(context, axis=1)
        
    def get_config(self):
        return super().get_config()

# ======================
# 2. Model Building with Inference Mode
# ======================

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
    
    x = layers.Bidirectional(
            layers.LSTM(256, return_sequences=True, kernel_regularizer=regularizers.l2(0.001)))(x)
    x = layers.Bidirectional(
            layers.LSTM(128, return_sequences=True, kernel_regularizer=regularizers.l2(0.001)))(x)
    x = layers.Bidirectional(
            layers.LSTM(64, return_sequences=True, kernel_regularizer=regularizers.l2(0.001)))(x)
    
    context_vector = FixedAttentionBlock()(x)
    
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(context_vector)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(len(metadata['classes']), activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # Add inference mode control
    model.inference_mode = False
    model._original_dropout_rates = {}
    
    # Store original dropout rates
    for layer in model.layers:
        if hasattr(layer, 'dropout'):
            model._original_dropout_rates[layer.name] = layer.dropout.rate
    
    def set_inference_mode(model, mode):
        model.inference_mode = mode
        for layer in model.layers:
            if hasattr(layer, 'dropout'):
                layer.dropout.rate = 0.0 if mode else model._original_dropout_rates[layer.name]
    
    model.set_inference_mode = lambda mode: set_inference_mode(model, mode)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ======================
# 3. Save/Load Model
# ======================

def save_model_components(model, save_dir):
    """Save complete model (architecture + weights)"""
    os.makedirs(save_dir, exist_ok=True)
    
    model.save(os.path.join(save_dir, 'full_model.keras'))
    
    model_json = model.to_json()
    with open(os.path.join(save_dir, 'model_architecture.json'), 'w', encoding='utf-8') as f:
        f.write(model_json)
    
    model.save_weights(os.path.join(save_dir, 'model_weights.weights.h5'))
    
    print(f"Model components saved to: {save_dir}")

def rebuild_model(metadata, components_dir):
    """Rebuild model from saved components"""
    try:
        model = tf.keras.models.load_model(
            os.path.join(components_dir, 'full_model.keras'),
            custom_objects={
                'FixedAttentionBlock': FixedAttentionBlock,
                'residual_block': residual_block
            }
        )
        # Restore dropout rates
        if hasattr(model, '_original_dropout_rates'):
            for layer in model.layers:
                if layer.name in model._original_dropout_rates:
                    layer.dropout.rate = model._original_dropout_rates[layer.name]
        print("Loaded full model directly")
        return model
    except Exception as e:
        print(f"Failed to load full model: {str(e)}")
    
    try:
        with open(os.path.join(components_dir, 'model_architecture.json'), 'r', encoding='utf-8') as f:
            model = models.model_from_json(
                f.read(),
                custom_objects={
                    'FixedAttentionBlock': FixedAttentionBlock,
                    'residual_block': residual_block
                }
            )
        
        model.load_weights(os.path.join(components_dir, 'model_weights.weights.h5'))
        
        # Add inference mode control to rebuilt model
        model.inference_mode = False
        model._original_dropout_rates = {}
        for layer in model.layers:
            if hasattr(layer, 'dropout'):
                model._original_dropout_rates[layer.name] = layer.dropout.rate
        
        def set_inference_mode(model, mode):
            model.inference_mode = mode
            for layer in model.layers:
                if hasattr(layer, 'dropout'):
                    layer.dropout.rate = 0.0 if mode else model._original_dropout_rates[layer.name]
        
        model.set_inference_mode = lambda mode: set_inference_mode(model, mode)
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print("Rebuilt model from architecture + weights")
        return model
    except Exception as e:
        print(f"Failed to rebuild model: {str(e)}")
        return None

# ======================
# 4. Feature Extraction
# ======================

def extract_features(audio_path, max_frames, n_features):
    """Deterministic feature extraction pipeline"""
    target_sr = 22050
    frame_length = 2048
    hop_length = 512
    n_mels = 128
    
    audio, sr = librosa.load(audio_path, sr=target_sr)
    
    # Extract features with fixed parameters
    mfcc = librosa.feature.mfcc(y=audio, sr=target_sr, n_mfcc=13,
                               n_fft=frame_length, hop_length=hop_length)
    chroma = librosa.feature.chroma_stft(y=audio, sr=target_sr,
                                      n_fft=frame_length, hop_length=hop_length)
    mel = librosa.feature.melspectrogram(y=audio, sr=target_sr, n_mels=n_mels,
                                       n_fft=frame_length, hop_length=hop_length)
    mel = librosa.power_to_db(mel, ref=np.max)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    
    # Combine features (adjust based on n_features)
    features = np.vstack([
        mfcc,          # 13
        delta_mfcc,    # 13
        delta2_mfcc,   # 13
        chroma,       # 12
        mel[:n_features-13*3-12]  # Remaining features from mel
    ])
    
    # Pad/truncate time dimension
    if features.shape[1] < max_frames:
        pad_width = ((0, 0), (0, max_frames - features.shape[1]))
        features = np.pad(features, pad_width, mode='constant')
    else:
        features = features[:, :max_frames]
    
    return np.expand_dims(features.T, axis=(0, -1)).astype(np.float32)

# ======================
# 5. Verification with Stability Checks
# ======================

def verify_model(model, metadata, test_file_path):
    """Verify model with stability checks"""
    try:
        # Configure for deterministic behavior
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
        
        print(f"\nTesting file: {Path(test_file_path).name}")
        
        # Extract features once
        features = extract_features(test_file_path, 
                                  metadata['max_frames'], 
                                  metadata['n_features'])
        
        # Run multiple predictions to check stability
        print("\nRunning stability checks (5 identical predictions):")
        predictions = []
        for i in range(5):
            pred = model.predict(features, verbose=0)[0]
            pred_class = metadata['classes'][np.argmax(pred)]
            predictions.append((pred_class, pred))
            print(f"Attempt {i+1}: {pred_class} (confidence: {np.max(pred):.4f})")
        
        # Check consistency
        unique_preds = len(set([p[0] for p in predictions]))
        if unique_preds > 1:
            print(f"\nWARNING: Found {unique_preds} different predictions!")
            print("Possible issues:")
            print("- Dropout still active during inference")
            print("- Randomness in feature extraction")
            print("- Numerical instability")
        else:
            print("\nPredictions are stable")
        
        # Return first prediction and all probabilities
        return predictions[0][0], predictions[0][1]
        
    except Exception as e:
        print(f"\nVerification failed: {str(e)}")
        return None, None

# ======================
# 6. Main Execution
# ======================

if __name__ == "__main__":
    # Example metadata format (replace with your actual loading code)
    # metadata = {
    #     'max_frames': max_frames,  # From your data
    #     'n_features': n_features,  # From your feature extraction
    #     'classes': df['label'].unique().tolist()  # Your actual class names
    # }
    
    # For testing purposes, using example values
    metadata = {
        'max_frames': 3826,
        'n_features': 145,
        'classes': ['Happy', 'Sad', 'Frustrated', 'Neutral', 'Angry']
    }
    
    save_dir = r"C:\Users\cmanw\OneDrive\Documents\Emotion-Recognition\Thai-Speech-Emotion-Recognition\Backend\model_components"
    test_files = [
        r"C:\Users\cmanw\OneDrive\Documents\Asset\Balanced_audio\Happy\s001_con_actor001_impro2_17.flac",
        r"C:\Users\cmanw\OneDrive\Documents\Asset\Balanced_audio\Happy\s001_con_actor001_impro2_21.flac",
        r"C:\Users\cmanw\OneDrive\Documents\Asset\Balanced_audio\Happy\s001_con_actor001_impro2_22.flac"
    ]
    
    try:
        # Rebuild model
        print("Loading model...")
        model = rebuild_model(metadata, save_dir)
        
        if model:
            # Set to inference mode (disables dropout)
            model.set_inference_mode(True)
            
            # Test each file
            for test_file in test_files:
                if not Path(test_file).exists():
                    print(f"\nFile not found: {test_file}")
                    continue
                
                # Verify model with stability checks
                pred_class, probs = verify_model(model, metadata, test_file)
                
                if pred_class:
                    print(f"\nFinal prediction for {Path(test_file).name}:")
                    print(f"Class: {pred_class}")
                    print("Probabilities:")
                    for emo, prob in zip(metadata['classes'], probs):
                        print(f"{emo}: {prob:.4f}")
                else:
                    print("\nPrediction failed")
        else:
            print("Failed to load model")
            
    except Exception as e:
        print(f"\nFatal error: {str(e)}")