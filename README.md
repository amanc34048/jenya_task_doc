# Generative AI with LSTM - Text Generation

An end-to-end deep learning project that builds, trains, regularizes, and evaluates a **Generative AI Text Generator using Long Short-Term Memory (LSTM)** neural networks in TensorFlow/Keras.

This repository fulfills all primary and bonus requirements specified in the **Generative AI Engineer Task**:
- Automated dataset downloading, cleaning, and sliding-window $(X, y)$ sequence extraction.
- Modular architecture implementation supporting both **Single-Layer Baseline** and **Deeper 2-Layer Stacked LSTM** with dropout regularization.
- Robust training loop featuring `ModelCheckpoint`, `EarlyStopping`, and adaptive learning rate decay (`ReduceLROnPlateau`).
- Autoregressive text generation with **Temperature-Controlled Softmax Sampling** across diverse seed prompts.
- Deliverables: complete Python source modules, a runnable end-to-end script (`train_and_generate.py`), an interactive Jupyter Notebook (`lstm_text_generation.ipynb`), sample outputs (`generated_samples.txt`), and architecture comparison analysis.

---

## 1. Project Structure

```text
├── data_preprocessing.py      # Dataset download, cleaning, tokenization & tf.data pipeline
├── model.py                   # LSTM neural network architectures (Baseline & Deeper)
├── generate.py                # Autoregressive generation & temperature sampling logic
├── train_and_generate.py      # End-to-end training, checkpointing & sample generation script
├── lstm_text_generation.ipynb # Interactive Jupyter Notebook with walkthrough & plots
├── generated_samples.txt      # Model outputs generated from multiple seeds and temperatures
├── baseline_lstm.keras        # Saved checkpoint for Single-Layer Baseline model
├── deep_lstm.keras            # Saved checkpoint for Deeper 2-Layer Stacked LSTM
├── shakespeare.txt            # Corpus text file (automatically downloaded)
└── README.md                  # Comprehensive project documentation
```

---

## 2. Dataset Information

- **Dataset**: *The Complete Works of William Shakespeare / Tiny Shakespeare*
- **Source**: [Project Gutenberg / Tiny Shakespeare (Andrej Karpathy char-rnn)](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)
- **Format**: Plain text (`.txt`), ~1.11 MB, 1,115,394 characters.
- **Automated Download**: The pipeline automatically downloads and caches `shakespeare.txt` if not already present on disk.

---

## 3. Preprocessing Pipeline (`data_preprocessing.py`)

1. **Text Normalization**:
   - Conversion to lowercase to standardize the vocabulary.
   - Punctuation normalization preserving syntactic flow for poetic and theatrical cadence.
2. **Tokenization (`TextTokenizer`)**:
   - Character-level vocabulary mapping (`token_to_idx` and `idx_to_token`).
   - Compact vocabulary size (~37 unique characters) ensuring high sample density and zero out-of-vocabulary issues during sampling.
3. **Sliding-Window Sequence Extraction**:
   - Sliding window of length `seq_length = 40` with stride `step = 3`.
   - Each input $X_i$ is a sequence of 40 token IDs; target $y_i$ is the 41st token ID.
4. **Data Pipeline (`tf.data.Dataset`)**:
   - 80% Training / 20% Validation split.
   - Batched with `batch_size = 128`, shuffled with buffer size 10,000, and prefetched via `tf.data.AUTOTUNE` for zero GPU/CPU starvation.

---

## 4. Model Architectures (`model.py`)

### Architecture 1: Baseline Single-Layer LSTM
- **Embedding Layer**: Vocabulary size $\to$ 64 dense dimensions.
- **LSTM Layer 1**: 256 hidden units.
- **Dropout**: Rate = 0.2 (prevents co-adaptation of recurrent units).
- **Dense Output Layer**: Softmax activation over vocabulary size.
- **Loss**: `SparseCategoricalCrossentropy`
- **Optimizer**: `Adam(learning_rate=0.003)`

### Architecture 2 (Bonus): Deeper 2-Layer Stacked LSTM
- **Embedding Layer**: Vocabulary size $\to$ 64 dense dimensions.
- **LSTM Layer 1**: 256 hidden units with `return_sequences=True`.
- **Dropout 1**: Rate = 0.2.
- **LSTM Layer 2**: 256 hidden units with `return_sequences=False`.
- **Dropout 2**: Rate = 0.2.
- **Dense Output Layer**: Softmax activation over vocabulary size.
- **Parameters**: ~1.05M trainable parameters providing higher representational capacity for capturing hierarchical grammatical structures.

---

## 5. Training & Regularization

Training is governed by three defensive callbacks to prevent overfitting and ensure fast convergence:
1. **`ModelCheckpoint`**: Monitors `val_loss` and persists only the best model weights (`.keras` format).
2. **`EarlyStopping`**: Halts training if `val_loss` fails to improve for 3 consecutive epochs and restores the best model weights.
3. **`ReduceLROnPlateau`**: Halves the learning rate (factor=0.5) when validation loss plateaus for 2 epochs, allowing fine-grained convergence.

---

## 6. Autoregressive Text Generation & Temperature Sampling (`generate.py`)

During inference, given an arbitrary seed prompt $S = (t_1, t_2, \dots, t_k)$:
1. The seed is tokenized, left-padded (or truncated) to `seq_length = 40`.
2. The model outputs unnormalized logits $z_i$.
3. **Temperature Scaling**: Softmax probabilities are modulated by temperature parameter $T$:
   $$P(w_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$
   - **$T = 0.2$ (Low Entropy)**: Highly confident, conservative, grammatical, but tends to repeat common phrases.
   - **$T = 0.5$ (Balanced)**: Good syntactic coherence, varied vocabulary, strong Shakespearean prose structure.
   - **$T = 0.8$ (Creative)**: Rich poetic vocabulary, authentic character dialogue patterns.
   - **$T = 1.2$ (High Entropy)**: Highly adventurous and diverse, with occasional phonetic coinages.
4. Next token is sampled multinomial-fashion from $P(w)$, appended to the sequence, and the sliding window advances by 1 token iteratively.

---

## 7. How to Run

### Requirements
```bash
pip install tensorflow keras numpy
```

### Running the Complete Pipeline
```bash
# Execute training and sample generation
python train_and_generate.py --epochs 6 --batch_size 128 --seq_length 40
```

### Command Line Options
- `--epochs`: Maximum number of training epochs (default: `12`).
- `--batch_size`: Mini-batch size (default: `128`).
- `--seq_length`: Input sequence context length (default: `40`).
- `--step`: Stride for sequence extraction (default: `3`).
- `--max_chars`: Maximum characters of corpus to process (default: `150000`).
- `--run_bonus`: Include deeper 2-layer LSTM experiment (default: `True`).

### Running the Jupyter Notebook
Launch Jupyter and open [`lstm_text_generation.ipynb`](file:///e:/DataScience/task/lstm_text_generation.ipynb) to inspect step-by-step visualizations, training curves, and interactive generation widgets.
