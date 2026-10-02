"""
Text Generation Module for Generative AI LSTM.

Implements iterative autoregressive text generation, seed sequence preprocessing,
and temperature-controlled probability sampling.
"""

import os
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


def sample_with_temperature(probabilities: np.ndarray, temperature: float = 1.0) -> int:
    """
    Applies temperature scaling to probabilities and samples a token index.

    - Low temperature (T < 0.5): High confidence, conservative, repetitive text.
    - Medium temperature (T ~ 0.7): Balanced creativity and coherence.
    - High temperature (T >= 1.0): High diversity, creative, potentially disordered text.
    """
    if temperature <= 1e-4:
        # Pure greedy (argmax)
        return int(np.argmax(probabilities))

    # Convert to float64 to avoid numerical instability
    probs = np.asarray(probabilities).astype("float64")
    # Clip probabilities to avoid log(0)
    probs = np.clip(probs, 1e-10, 1.0)
    log_probs = np.log(probs) / temperature
    exp_probs = np.exp(log_probs)
    scaled_probs = exp_probs / np.sum(exp_probs)

    # Multinomial sampling
    sampled_indices = np.random.multinomial(1, scaled_probs, 1)
    return int(np.argmax(sampled_indices))


def generate_text(
    model: tf.keras.Model,
    tokenizer,
    seed_text: str,
    num_generate: int = 300,
    temperature: float = 0.7,
    seq_length: int = 40
) -> str:
    """
    Generates text starting from a seed string.

    Args:
        model: Trained Keras LSTM model.
        tokenizer: TextTokenizer instance used during training.
        seed_text: Initial seed text prompt.
        num_generate: Number of characters/tokens to generate.
        temperature: Temperature factor for sampling randomness.
        seq_length: Input sequence window length expected by the model.

    Returns:
        The generated text string including the seed.
    """
    # Clean and tokenize seed
    cleaned_seed = seed_text.lower()
    encoded_seed = tokenizer.encode(cleaned_seed)

    # Ensure seed is at least seq_length by left-padding or truncating
    if len(encoded_seed) < seq_length:
        pad_token = tokenizer.encode(" ")[0] if " " in tokenizer.token_to_idx else 0
        current_input = [pad_token] * (seq_length - len(encoded_seed)) + encoded_seed
    else:
        current_input = encoded_seed[-seq_length:]

    # Safe index clamping against model embedding input_dim
    vocab_limit = getattr(model.layers[0], "input_dim", None)
    if vocab_limit is not None:
        current_input = [min(max(0, int(idx)), vocab_limit - 1) for idx in current_input]

    generated_tokens = []

    for _ in range(num_generate):
        # Bound input indices
        if vocab_limit is not None:
            bounded_input = [min(max(0, int(idx)), vocab_limit - 1) for idx in current_input]
        else:
            bounded_input = current_input

        # Shape: (1, seq_length)
        input_tensor = tf.constant([bounded_input], dtype=tf.int32)
        predictions = model(input_tensor, training=False).numpy()[0]

        next_idx = sample_with_temperature(predictions, temperature=temperature)

        generated_tokens.append(next_idx)
        # Shift sliding window: remove first token, append predicted token
        current_input = current_input[1:] + [next_idx]

    generated_text = tokenizer.decode(generated_tokens)
    return seed_text + generated_text


def format_generation_report(model, tokenizer, seed_prompts, temperatures=[0.2, 0.5, 0.8, 1.0], num_generate=250, seq_length=40):
    """
    Runs text generation across multiple seeds and temperatures,
    returning a formatted string report.
    """
    report_lines = []
    separator = "=" * 80
    report_lines.append(separator)
    report_lines.append(f"TEXT GENERATION REPORT (Model: {model.name})")
    report_lines.append(separator)

    for seed in seed_prompts:
        report_lines.append(f"\n>>> SEED PROMPT: \"{seed}\"")
        report_lines.append("-" * 60)
        for temp in temperatures:
            output = generate_text(
                model=model,
                tokenizer=tokenizer,
                seed_text=seed,
                num_generate=num_generate,
                temperature=temp,
                seq_length=seq_length
            )
            report_lines.append(f"\n[Temperature: {temp:.1f}]")
            report_lines.append(output.strip())
            report_lines.append("-" * 40)

    report_lines.append("\n" + separator + "\n")
    return "\n".join(report_lines)


if __name__ == "__main__":
    import argparse
    from data_preprocessing import download_dataset, load_and_clean_text, TextTokenizer

    parser = argparse.ArgumentParser(description="Generate text using a trained LSTM model.")
    parser.add_argument("--model", type=str, default="baseline_lstm.keras", help="Path to .keras model checkpoint")
    parser.add_argument("--seed", type=str, default="to be, or not to be, that is the question: ", help="Seed text prompt")
    parser.add_argument("--length", type=int, default=200, help="Number of characters to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (e.g. 0.2, 0.5, 0.8, 1.2)")
    parser.add_argument("--seq_length", type=int, default=40, help="Sliding window sequence length")
    args = parser.parse_args()

    model_file = args.model
    if not os.path.exists(model_file):
        if os.path.exists("deep_lstm.keras"):
            model_file = "deep_lstm.keras"
        else:
            raise FileNotFoundError(f"Model file '{args.model}' not found. Run train_and_generate.py first.")

    print("=" * 60)
    print(" GENERATIVE AI LSTM TEXT GENERATOR")
    print("=" * 60)
    print(f"Loading model checkpoint: {model_file}")
    loaded_model = tf.keras.models.load_model(model_file)

    # Initialize tokenizer with the vocabulary size matching the loaded model
    data_path = download_dataset()
    vocab_dim = getattr(loaded_model.layers[0], "input_dim", 37)
    char_limit = 50000 if vocab_dim == 36 else 150000
    corpus = load_and_clean_text(data_path, lowercase=True, max_chars=char_limit)
    tokenizer = TextTokenizer(level="char")
    tokenizer.fit(corpus)

    print(f"\nSeed Prompt: \"{args.seed}\"")
    print(f"Temperature: {args.temperature} | Length: {args.length} characters")
    print("-" * 60)
    generated_output = generate_text(
        model=loaded_model,
        tokenizer=tokenizer,
        seed_text=args.seed,
        num_generate=args.length,
        temperature=args.temperature,
        seq_length=args.seq_length
    )
    print(generated_output)
    print("=" * 60)

