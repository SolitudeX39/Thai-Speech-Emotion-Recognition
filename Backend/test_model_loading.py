import os
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import numpy as np
from pathlib import Path

# ======================
# 1. Define Model Components
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
# 2. Model Building
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
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ======================
# 3. Save Architecture and Weights
# ======================

def save_model_components(model, save_dir):
    """Save complete model (architecture + weights)"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save full model (architecture + weights + optimizer state)
    model.save(os.path.join(save_dir, 'full_model.keras'))
    
    # Save just architecture (JSON)
    model_json = model.to_json()
    with open(os.path.join(save_dir, 'model_architecture.json'), 'w') as f:
        f.write(model_json)
    
    # Save just weights
    model.save_weights(os.path.join(save_dir, 'model_weights.weights.h5'))
    
    print(f"Model components saved to: {save_dir}")

# ======================
# 4. Rebuild Model
# ======================

def rebuild_model(metadata, components_dir):
    """Rebuild model from saved components"""
    # Method 1: Load full model directly
    try:
        model = tf.keras.models.load_model(
            os.path.join(components_dir, 'full_model.keras'),
            custom_objects={
                'FixedAttentionBlock': FixedAttentionBlock,
                'residual_block': residual_block
            }
        )
        print("Loaded full model directly")
        return model
    except:
        pass
    
    # Method 2: Rebuild from architecture + weights
    try:
        # Load architecture
        with open(os.path.join(components_dir, 'model_architecture.json'), 'r') as f:
            model = models.model_from_json(
                f.read(),
                custom_objects={
                    'FixedAttentionBlock': FixedAttentionBlock,
                    'residual_block': residual_block
                }
            )
        
        # Load weights
        model.load_weights(os.path.join(components_dir, 'model_weights.weights.h5'))
        
        # Recompile
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print("Rebuilt model from architecture + weights")
        return model
    except Exception as e:
        print(f"Failed to rebuild model: {e}")
        return None

# ======================
# 5. Verification
# ======================

def verify_model(model, metadata):
    """Verify model structure and performance"""
    print("\n=== Model Summary ===")
    model.summary()
    
    print("\n=== Layer Verification ===")
    print(f"Total layers: {len(model.layers)}")
    
    print("\n=== Shape Verification ===")
    dummy_input = np.random.rand(1, metadata['max_frames'], metadata['n_features'], 1)
    try:
        pred = model.predict(dummy_input)
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {pred.shape}")
        print(f"Matches classes: {pred.shape[1] == len(metadata['classes'])}")
    except Exception as e:
        print(f"Prediction failed: {e}")

# ======================
# 6. Main Execution
# ======================

if __name__ == "__main__":
    # Configuration
    metadata = {
        'max_frames': 3826,
        'n_features': 145,
        'classes': ['happy', 'sad', 'angry', 'neutral', 'surprise']
    }
    save_dir = r"C:\Users\cmanw\OneDrive\Documents\Emotion-Recognition\Thai-Speech-Emotion-Recognition\Backend\model_components"
    
    # 1. Build model
    print("Building model...")
    model = build_hybrid_model(metadata)
    
    # 2. Save components
    print("\nSaving model components...")
    save_model_components(model, save_dir)
    
    # 3. Rebuild model
    print("\nRebuilding model...")
    rebuilt_model = rebuild_model(metadata, save_dir)
    
    # 4. Verify
    if rebuilt_model:
        print("\nVerifying rebuilt model...")
        verify_model(rebuilt_model, metadata)