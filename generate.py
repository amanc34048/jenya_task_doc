"""
Text Generation Module for Generative AI LSTM.

Implements iterative autoregressive text generation, seed sequence preprocessing,
and temperature-controlled probability sampling.
"""

import numpy as np

# Ensure compatibility with SciPy/NumPy versions
if not hasattr(np, "long"):
    np.long = int
if not hasattr(np, "ulong"):
    np.ulong = int

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
