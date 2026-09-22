"""
Model Design Module for Generative AI LSTM Text Generation.

Defines LSTM architectures (single-layer baseline and deeper multi-layer variants),
embedding layers, regularization (dropout), and model compilation.
"""

import numpy as np

# Ensure compatibility with SciPy/NumPy versions
if not hasattr(np, "long"):
    np.long = int
if not hasattr(np, "ulong"):
    np.ulong = int

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


def build_lstm_model(
    vocab_size: int,
    embedding_dim: int = 64,
    rnn_units: int = 256,
    num_layers: int = 1,
    dropout_rate: float = 0.2,
    learning_rate: float = 0.002,
    model_name: str = "LSTM_Text_Generator"
) -> tf.keras.Model:
    """
    Builds and compiles an LSTM-based text generation model.

    Architecture:
    1. Input layer (implicit sequence of integers)
    2. Embedding layer: maps token indices to dense vectors
    3. One or more LSTM layers (with return_sequences=True for intermediate layers)
    4. Dropout layers: prevent overfitting
    5. Dense output layer: size=vocab_size with Softmax activation
    """
    model = models.Sequential(name=model_name)

    # 1. Embedding Layer
    model.add(layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        mask_zero=False,
        name="token_embedding"
    ))

    # 2. LSTM Layers
    if num_layers == 1:
        model.add(layers.LSTM(
            rnn_units,
            return_sequences=False,
            name="lstm_layer_1"
        ))
        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate, name="dropout_1"))
    else:
        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            model.add(layers.LSTM(
                rnn_units,
                return_sequences=not is_last,
                name=f"lstm_layer_{i+1}"
            ))
            if dropout_rate > 0:
                model.add(layers.Dropout(dropout_rate, name=f"dropout_{i+1}"))

    # 3. Dense Output Layer with Softmax
    model.add(layers.Dense(vocab_size, activation="softmax", name="token_prediction"))

    # 4. Optimizer & Loss compilation
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    sample_model = build_lstm_model(vocab_size=65, num_layers=2)
    sample_model.summary()
