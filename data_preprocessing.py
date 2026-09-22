"""
Data Preprocessing Module for Generative AI LSTM Text Generation.

Handles downloading, cleaning, tokenizing (character-level and word-level),
and preparing sliding-window input-output sequences for training.
"""

import os
import re
import urllib.request
import numpy as np

# Ensure compatibility with SciPy/NumPy versions
if not hasattr(np, "long"):
    np.long = int
if not hasattr(np, "ulong"):
    np.ulong = int

import tensorflow as tf

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATASET_PATH = "shakespeare.txt"


def download_dataset(url: str = SHAKESPEARE_URL, destination: str = DATASET_PATH) -> str:
    """
    Downloads the text dataset if not already present.
    """
    if not os.path.exists(destination):
        print(f"Downloading dataset from {url} to {destination}...")
        urllib.request.urlretrieve(url, destination)
        print(f"Download complete: {destination} ({os.path.getsize(destination)} bytes)")
    else:
        print(f"Dataset already present at: {destination} ({os.path.getsize(destination)} bytes)")
    return destination


def load_and_clean_text(filepath: str, lowercase: bool = True, remove_punctuation: bool = False, max_chars: int = None) -> str:
    """
    Loads text from file and applies cleaning steps:
    - Lowercase conversion (optional)
    - Punctuation removal or normalization (optional)
    """
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    if max_chars is not None and len(text) > max_chars:
        text = text[:max_chars]

    if lowercase:
        text = text.lower()

    if remove_punctuation:
        # Keep alphanumeric, whitespace, and basic punctuation
        text = re.sub(r"[^\w\s]", "", text)

    return text


class TextTokenizer:
    """
    Handles character-level or word-level mapping to and from integer indices.
    """
    def __init__(self, level: str = "char"):
        assert level in ("char", "word"), "level must be 'char' or 'word'"
        self.level = level
        self.token_to_idx = {}
        self.idx_to_token = {}
        self.vocab = []
        self.vocab_size = 0

    def fit(self, text: str):
        if self.level == "char":
            tokens = sorted(list(set(text)))
        else:
            tokens = sorted(list(set(text.split())))

        self.vocab = tokens
        self.vocab_size = len(tokens)
        self.token_to_idx = {tok: idx for idx, tok in enumerate(tokens)}
        self.idx_to_token = {idx: tok for idx, tok in enumerate(tokens)}
        print(f"[{self.level.upper()} Tokenizer] Vocabulary size: {self.vocab_size}")

    def tokenize(self, text: str):
        if self.level == "char":
            return list(text)
        return text.split()

    def encode(self, sequence):
        """Converts tokens to indices. Handles unknown tokens using a fallback."""
        default_idx = 0
        if isinstance(sequence, str):
            tokens = self.tokenize(sequence)
        else:
            tokens = sequence
        return [self.token_to_idx.get(t, default_idx) for t in tokens]

    def decode(self, indices):
        """Converts indices back to tokens/text."""
        tokens = [self.idx_to_token.get(idx, "") for idx in indices]
        if self.level == "char":
            return "".join(tokens)
        return " ".join(tokens)


def create_sequences(encoded_text, seq_length: int = 40, step: int = 3):
    """
    Creates input-output pairs using a sliding window:
    - Input (X): sequence of tokens of length `seq_length`
    - Target (y): next token immediately following the input sequence
    """
    sentences = []
    next_tokens = []
    for i in range(0, len(encoded_text) - seq_length, step):
        sentences.append(encoded_text[i : i + seq_length])
        next_tokens.append(encoded_text[i + seq_length])

    X = np.array(sentences, dtype=np.int32)
    y = np.array(next_tokens, dtype=np.int32)
    print(f"Created {len(X)} sequence pairs (seq_length={seq_length}, step={step}).")
    return X, y


def prepare_tf_dataset(X, y, val_split: float = 0.2, batch_size: int = 128, shuffle: bool = True):
    """
    Splits into training and validation sets, returning efficient tf.data.Dataset pipelines.
    """
    total_samples = len(X)
    val_count = int(total_samples * val_split)
    train_count = total_samples - val_count

    indices = np.arange(total_samples)
    if shuffle:
        np.random.seed(42)
        np.random.shuffle(indices)

    train_idx = indices[:train_count]
    val_idx = indices[train_count:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))

    if shuffle:
        train_ds = train_ds.shuffle(buffer_size=10000, seed=42)

    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    print(f"Dataset split: Train samples = {train_count}, Val samples = {val_count}, Batch size = {batch_size}")
    return train_ds, val_ds, (X_train, y_train), (X_val, y_val)
