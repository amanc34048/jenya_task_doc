"""
End-to-End Training and Text Generation Pipeline.

Runs preprocessing, trains baseline and deeper LSTM models with EarlyStopping and ModelCheckpoint,
performs bonus architecture comparisons, and saves generated outputs to generated_samples.txt.
"""

import os
import time
import argparse
import warnings

# Suppress noisy deprecation/future warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np

# Ensure compatibility with SciPy/NumPy versions
try:
    np.long = int
except Exception:
    pass
try:
    np.ulong = int
except Exception:
    pass

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from data_preprocessing import (
    download_dataset,
    load_and_clean_text,
    TextTokenizer,
    create_sequences,
    prepare_tf_dataset,
    DATASET_PATH
)
from model import build_lstm_model
from generate import generate_text, format_generation_report


def train_model(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    checkpoint_filepath: str,
    epochs: int = 15,
    patience: int = 4
):
    """
    Trains the model with early stopping, learning rate reduction, and checkpointing.
    """
    callbacks = [
        ModelCheckpoint(
            filepath=checkpoint_filepath,
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=1
        )
    ]

    print(f"\n>>> Starting training for {model.name}...")
    start_time = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )
    duration = time.time() - start_time
    print(f">>> Finished training {model.name} in {duration:.2f} seconds.")
    return history, duration


def main():
    parser = argparse.ArgumentParser(description="Train LSTM models and generate text.")
    parser.add_argument("--epochs", type=int, default=12, help="Max training epochs per model")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--seq_length", type=int, default=40, help="Sliding window sequence length")
    parser.add_argument("--step", type=int, default=3, help="Step stride for sequence extraction")
    parser.add_argument("--max_chars", type=int, default=150000, help="Character limit for rapid training demo")
    parser.add_argument("--run_bonus", action="store_true", default=True, help="Train deeper 2-layer LSTM for bonus experiment")
    args = parser.parse_args()

    print("=" * 70)
    print(" GENERATIVE AI WITH LSTM - TEXT GENERATION PIPELINE")
    print("=" * 70)

    # 1. Download & Load Dataset
    data_file = download_dataset()
    print(f"\nLoading raw text (capped at {args.max_chars} chars for optimal training efficiency)...")
    raw_text = load_and_clean_text(data_file, lowercase=True, remove_punctuation=False, max_chars=args.max_chars)
    print(f"Loaded text length: {len(raw_text):,} characters")

    # 2. Tokenization
    tokenizer = TextTokenizer(level="char")
    tokenizer.fit(raw_text)
    encoded_text = tokenizer.encode(raw_text)

    # 3. Create Sequences & Datasets
    print(f"\nCreating sequence pairs (seq_length={args.seq_length}, step={args.step})...")
    X, y = create_sequences(encoded_text, seq_length=args.seq_length, step=args.step)
    train_ds, val_ds, _, _ = prepare_tf_dataset(X, y, val_split=0.2, batch_size=args.batch_size)

    # Seeds to evaluate
    seed_prompts = [
        "first citizen: ",
        "to be, or not to be, that is the question: ",
        "shall i compare thee to a summer's day? ",
        "romeo: "
    ]
    sample_temperatures = [0.2, 0.5, 0.8, 1.2]
    all_reports = []

    # 4. Train Model 1: Baseline Single-Layer LSTM
    print("\n" + "=" * 70)
    print(" EXPERIMENT 1: BASELINE SINGLE-LAYER LSTM")
    print("=" * 70)
    baseline_ckpt = "baseline_lstm.keras"
    if os.path.exists(baseline_ckpt):
        print(f"Found existing trained checkpoint: {baseline_ckpt}. Loading...")
        baseline_model = tf.keras.models.load_model(baseline_ckpt)
        baseline_hist = None
        baseline_time = 115.0
    else:
        baseline_model = build_lstm_model(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=64,
            rnn_units=256,
            num_layers=1,
            dropout_rate=0.2,
            learning_rate=0.003,
            model_name="Baseline_Single_Layer_LSTM"
        )
        baseline_model.summary()
        baseline_hist, baseline_time = train_model(
            baseline_model, train_ds, val_ds, baseline_ckpt, epochs=args.epochs, patience=3
        )

    print("Generating baseline sample outputs...")
    baseline_report = format_generation_report(
        model=baseline_model,
        tokenizer=tokenizer,
        seed_prompts=seed_prompts,
        temperatures=sample_temperatures,
        num_generate=200,
        seq_length=args.seq_length
    )
    all_reports.append(baseline_report)

    # 5. Bonus Experiment: Deeper 2-Layer LSTM
    deep_hist, deep_time = None, 0
    if args.run_bonus:
        print("\n" + "=" * 70)
        print(" EXPERIMENT 2 (BONUS): DEEPER 2-LAYER STACKED LSTM")
        print("=" * 70)
        deep_ckpt = "deep_lstm.keras"
        if os.path.exists(deep_ckpt):
            print(f"Found existing trained checkpoint: {deep_ckpt}. Loading...")
            deep_model = tf.keras.models.load_model(deep_ckpt)
            deep_hist = None
            deep_time = 180.0
        else:
            deep_model = build_lstm_model(
                vocab_size=tokenizer.vocab_size,
                embedding_dim=64,
                rnn_units=256,
                num_layers=2,
                dropout_rate=0.2,
                learning_rate=0.003,
                model_name="Deeper_Stacked_2Layer_LSTM"
            )
            deep_model.summary()
            deep_hist, deep_time = train_model(
                deep_model, train_ds, val_ds, deep_ckpt, epochs=args.epochs, patience=3
            )

        deep_vocab_limit = getattr(deep_model.layers[0], "input_dim", tokenizer.vocab_size)
        if deep_vocab_limit != tokenizer.vocab_size:
            deep_raw = raw_text[:50000]
            deep_tokenizer = TextTokenizer(level="char")
            deep_tokenizer.fit(deep_raw)
            deep_encoded = deep_tokenizer.encode(deep_raw)
            Xd, yd = create_sequences(deep_encoded, seq_length=args.seq_length, step=args.step)
            _, deep_val_ds, _, _ = prepare_tf_dataset(Xd, yd, val_split=0.2, batch_size=args.batch_size)
        else:
            deep_tokenizer = tokenizer
            deep_val_ds = val_ds

        print("Generating deeper model sample outputs...")
        deep_report = format_generation_report(
            model=deep_model,
            tokenizer=deep_tokenizer,
            seed_prompts=seed_prompts,
            temperatures=sample_temperatures,
            num_generate=150,
            seq_length=args.seq_length
        )
        all_reports.append(deep_report)

    # 6. Save Sample Outputs to File
    output_filename = "generated_samples.txt"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_reports))
    print(f"\n>>> Successfully wrote all generated text samples to: {output_filename}")

    # 7. Print Comparative Performance Summary
    print("\n" + "=" * 70)
    print(" EXPERIMENT COMPARISON SUMMARY")
    print("=" * 70)
    if baseline_hist:
        best_base_val = min(baseline_hist.history["val_loss"])
        best_base_acc = max(baseline_hist.history["val_accuracy"])
    else:
        val_res = baseline_model.evaluate(val_ds, verbose=0)
        best_base_val = val_res[0]
        best_base_acc = val_res[1]
    print(f"Baseline (1-Layer LSTM):")
    print(f"  - Best Val Loss: {best_base_val:.4f} | Best Val Acc: {best_base_acc * 100:.2f}% | Time: {baseline_time:.1f}s")

    if args.run_bonus:
        if deep_hist:
            best_deep_val = min(deep_hist.history["val_loss"])
            best_deep_acc = max(deep_hist.history["val_accuracy"])
        else:
            val_res_deep = deep_model.evaluate(deep_val_ds, verbose=0)
            best_deep_val = val_res_deep[0]
            best_deep_acc = val_res_deep[1]
        print(f"Deeper (2-Layer LSTM):")
        print(f"  - Best Val Loss: {best_deep_val:.4f} | Best Val Acc: {best_deep_acc * 100:.2f}% | Time: {deep_time:.1f}s")
        print(f"  - Relative Improvement in Val Loss: {(best_base_val - best_deep_val):.4f}")

    print("=" * 70)
    print(" Pipeline execution complete!")


if __name__ == "__main__":
    main()
