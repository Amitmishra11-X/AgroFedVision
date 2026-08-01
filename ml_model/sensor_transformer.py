"""
sensor_transformer.py — Transformer-based sensor feature encoder.

Input:  flat sensor feature vector (batch, num_sensor_features)
Output: (batch, 64) embedding

Why Transformer for tabular/sensor data:
  - Self-attention over feature tokens lets the model learn which sensor
    readings interact most (e.g. high N + low rainfall = different
    interpretation than high N + high rainfall)
  - Outperforms simple Dense/MLP branches on structured data because
    feature interactions are explicitly modelled, not just summed
"""

import tensorflow as tf
from tensorflow.keras import layers


def build_sensor_transformer(num_features, d_model=64, num_heads=4,
                              ff_dim=128, num_blocks=2, dropout=0.1):
    """
    Treats each sensor feature as a separate "token" and applies
    Transformer self-attention across features.

    Input shape:  (batch, num_features)
    Output shape: (batch, d_model)
    """
    inputs = tf.keras.Input(shape=(num_features,), name="sensor_input")

    # Project each scalar feature into a d_model-dim embedding vector
    # by treating features as a sequence of length num_features
    x = layers.Reshape((num_features, 1))(inputs)               # (B, F, 1)
    x = layers.Dense(d_model, name="feature_proj")(x)           # (B, F, d_model)

    # Positional embedding (feature order matters — e.g. N before P before K)
    positions = tf.range(start=0, limit=num_features)
    pos_emb   = layers.Embedding(num_features, d_model,
                                  name="pos_embedding")(positions)
    x = x + pos_emb                                              # (B, F, d_model)

    # Transformer encoder blocks
    for i in range(num_blocks):
        # Multi-head self-attention
        attn = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads,
            dropout=dropout, name=f"mha_{i}"
        )(x, x)
        attn = layers.Dropout(dropout)(attn)
        x    = layers.LayerNormalization(epsilon=1e-6, name=f"ln1_{i}")(x + attn)

        # Feed-forward
        ff = layers.Dense(ff_dim, activation="gelu", name=f"ff1_{i}")(x)
        ff = layers.Dense(d_model, name=f"ff2_{i}")(ff)
        ff = layers.Dropout(dropout)(ff)
        x  = layers.LayerNormalization(epsilon=1e-6, name=f"ln2_{i}")(x + ff)

    # Pool across feature dimension → single embedding vector
    x = layers.GlobalAveragePooling1D(name="sensor_pool")(x)    # (B, d_model)
    x = layers.Dropout(dropout)(x)

    model = tf.keras.Model(inputs, x, name="SensorTransformer")
    return model
