"""
Multi-Model Architecture for Cyber Security Threat Detection.
Implements CNN-LSTM, Transformer, Autoencoder, and Ensemble models.
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten,
    BatchNormalization, Input, MultiHeadAttention, LayerNormalization,
    GlobalAveragePooling1D, Concatenate, Add, Bidirectional
)
from tensorflow.keras.optimizers import Adam
import numpy as np


# ============================================================
# 1. ENHANCED CNN-LSTM MODEL (with Attention)
# ============================================================
def build_cnn_lstm_model(input_shape, num_classes, learning_rate=0.001):
    """
    Enhanced CNN-LSTM with deeper layers, batch norm, and bidirectional LSTM.
    
    Args:
        input_shape: (timesteps, features)
        num_classes: Number of attack categories
        learning_rate: Adam optimizer LR
    Returns:
        Compiled Keras model
    """
    inputs = Input(shape=input_shape, name='cnn_lstm_input')
    
    # CNN Block 1
    x = Conv1D(filters=64, kernel_size=1, activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    
    # CNN Block 2
    x = Conv1D(filters=128, kernel_size=1, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    
    # CNN Block 3
    x = Conv1D(filters=256, kernel_size=1, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    
    # Bidirectional LSTM
    x = Bidirectional(LSTM(128, return_sequences=False))(x)
    x = Dropout(0.4)(x)
    
    # Dense layers
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    outputs = Dense(num_classes, activation='softmax', name='cnn_lstm_output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='CNN_LSTM')
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ============================================================
# 2. TRANSFORMER MODEL (Multi-Head Self-Attention)
# ============================================================
class TransformerBlock(tf.keras.layers.Layer):
    """Transformer encoder block with multi-head attention."""
    def __init__(self, d_model, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='relu'),
            Dense(d_model),
        ])
        self.ln1 = LayerNormalization(epsilon=1e-6)
        self.ln2 = LayerNormalization(epsilon=1e-6)
        self.drop1 = Dropout(dropout_rate)
        self.drop2 = Dropout(dropout_rate)
    
    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.drop1(attn_output, training=training)
        out1 = self.ln1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.drop2(ffn_output, training=training)
        return self.ln2(out1 + ffn_output)


def build_transformer_model(input_shape, num_classes, learning_rate=0.001):
    """
    Transformer-based classifier with multi-head self-attention.
    Captures complex feature interactions for sequence-aware detection.
    """
    inputs = Input(shape=input_shape, name='transformer_input')
    
    # Project to d_model dimension
    x = Dense(128)(inputs)
    
    # Transformer blocks
    x = TransformerBlock(d_model=128, num_heads=4, ff_dim=256, dropout_rate=0.2)(x)
    x = TransformerBlock(d_model=128, num_heads=4, ff_dim=256, dropout_rate=0.2)(x)
    
    # Global pooling
    x = GlobalAveragePooling1D()(x)
    
    # Classification head
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    outputs = Dense(num_classes, activation='softmax', name='transformer_output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='Transformer')
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ============================================================
# 3. AUTOENCODER (Unsupervised Anomaly Detection)
# ============================================================
def build_autoencoder_model(input_dim, encoding_dim=32, learning_rate=0.001):
    """
    Autoencoder for unsupervised anomaly detection.
    Trained on BENIGN traffic only; high reconstruction error = anomaly.
    
    Args:
        input_dim: Number of features
        encoding_dim: Bottleneck dimension
    """
    # Encoder
    inputs = Input(shape=(input_dim,), name='ae_input')
    x = Dense(128, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    encoded = Dense(encoding_dim, activation='relu', name='encoding')(x)
    
    # Decoder
    x = Dense(64, activation='relu')(encoded)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    decoded = Dense(input_dim, activation='linear', name='ae_output')(x)
    
    autoencoder = Model(inputs=inputs, outputs=decoded, name='Autoencoder')
    autoencoder.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    # Also create encoder-only model for feature extraction
    encoder = Model(inputs=inputs, outputs=encoded, name='Encoder')
    
    return autoencoder, encoder


# ============================================================
# 4. ENSEMBLE MODEL (Stacking CNN-LSTM + Transformer)
# ============================================================
def build_ensemble_model(cnn_lstm_model, transformer_model, num_classes, learning_rate=0.0005):
    """
    Ensemble model combining CNN-LSTM and Transformer predictions via stacking.
    Takes concatenated probability outputs from both models as input.
    """
    # Input: concatenated softmax probabilities from both models
    ensemble_input_dim = num_classes * 2  # from CNN-LSTM + Transformer
    
    inputs = Input(shape=(ensemble_input_dim,), name='ensemble_input')
    x = Dense(64, activation='relu')(inputs)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation='softmax', name='ensemble_output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='Ensemble')
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ============================================================
# FOCAL LOSS (for extreme class imbalance)
# ============================================================
class FocalLoss(tf.keras.losses.Loss):
    """
    Focal Loss for handling extreme class imbalance.
    Downweights easy examples and focuses on hard misclassifications.
    """
    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
    
    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=tf.shape(y_pred)[-1])
        cross_entropy = -y_true_one_hot * tf.math.log(y_pred)
        weight = self.alpha * y_true_one_hot * tf.pow(1.0 - y_pred, self.gamma)
        focal_loss = weight * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=-1))
