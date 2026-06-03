# LSTM Next-Word Predictor

A TensorFlow/Keras-based LSTM model for next-word prediction and text generation trained on Shakespeare's text corpus from Project Gutenberg.

## Overview

This project:

- Downloads and preprocesses Shakespeare's text corpus
- Tokenizes text into word sequences
- Trains a stacked LSTM neural network
- Predicts the next word in a sequence
- Generates text using temperature-controlled sampling

## Requirements

- Python
- TensorFlow
- NumPy
- Requests

Install dependencies:

```bash
pip install tensorflow numpy requests
```

## Dataset

The model automatically downloads Shakespeare's corpus from Project Gutenberg:

https://www.gutenberg.org/files/100/100-0.txt

Preprocessing steps:

- Convert text to lowercase
- Remove non-alphabetic characters
- Preserve apostrophes
- Collapse multiple spaces
- Limit corpus size to a configurable number of characters

## Model Architecture

```text
Embedding (100)
      ↓
LSTM (128, return_sequences=True)
      ↓
Dropout (0.3)
      ↓
LSTM (64)
      ↓
Dropout (0.3)
      ↓
Dense (Softmax)
```

### Hyperparameters

| Parameter | Value |
|------------|---------|
| Vocabulary Size | 10,000 |
| Sequence Length | 20 |
| Embedding Dimension | 100 |
| LSTM Units (Layer 1) | 128 |
| LSTM Units (Layer 2) | 64 |
| Dropout | 0.3 |
| Batch Size | 128 |
| Optimizer | Adam |
| Learning Rate | 0.001 |

## Training

The model uses:

- Sparse Categorical Crossentropy loss
- Early Stopping
- ReduceLROnPlateau
- Validation split from the dataset

After training, the model is saved as:

```text
lstm_next_word_safe.h5
```

## Text Generation

Example seed text:

```python
seed = "to be or not to be"
```

Generate text:

```python
generate_text(
    seed="to be or not to be",
    n_words=30,
    temperature=0.8
)
```

### Temperature Values

| Temperature | Behaviour |
|------------|------------|
| 0.3 | More predictable |
| 0.8 | Balanced |
| 1.2 | More creative |

## Output Metrics

After training, the script reports:

- Best Epoch
- Validation Loss
- Validation Accuracy
- Perplexity

## Learning Concepts

This project demonstrates:

- Natural Language Processing (NLP)
- Tokenization
- Sequence Modeling
- Word Prediction
- Recurrent Neural Networks (RNNs)
- Long Short-Term Memory Networks (LSTMs)
- Text Generation
