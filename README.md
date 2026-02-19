# Deep-Learning

1. Tensor Creation

1D, 2D, and 3D tensors

NumPy arrays vs PyTorch tensors

2. Basic Operations

Element-wise:

Addition

Subtraction

Multiplication

Division

3. Linear Algebra Operations

Dot product (1D vectors)

Matrix multiplication

4. Indexing and Slicing

Basic indexing

Row and column extraction

Boolean masking

Sub-tensor extraction

5. Tensor Reshaping

PyTorch:

.view()

.reshape()

.unsqueeze()

.squeeze()

NumPy:

.reshape()

Comparison between PyTorch and NumPy reshaping behavior

6. Broadcasting

Operations on tensors with different shapes

Broadcasting rules in NumPy and PyTorch

7. In-place vs Out-of-place Operations

Memory-efficient in-place operations

Risks with autograd in PyTorch

# EXPERIMENT 2
Experiment 2: Neural Network from Scratch using NumPy on MNIST Dataset
📌 Objective

To design and implement a feedforward neural network from scratch using NumPy (without using deep learning libraries such as TensorFlow or PyTorch) and train it on the MNIST handwritten digit dataset for digit classification.

📊 Dataset Description

The MNIST dataset consists of grayscale images of handwritten digits (0–9).

Image size: 28 × 28 pixels

Training samples: 60,000

Test samples: 10,000

Number of classes: 10

Each image is flattened into a 784-dimensional vector before being fed into the neural network.

🧠 Neural Network Architecture

The implemented neural network follows a fully connected feedforward architecture:

Layer	Description
Input Layer	784 neurons (28×28 pixels)
Hidden Layer	128 neurons
Output Layer	10 neurons (digits 0–9)
Activation Functions

ReLU – used in the hidden layer

Softmax – used in the output layer for multi-class classification

⚙️ Methodology

Load MNIST dataset from local IDX files

Normalize pixel values to range [0, 1]

One-hot encode class labels

Initialize weights and biases

Perform forward propagation

Compute classification error

Apply backpropagation to update parameters

Evaluate accuracy on test dataset after each epoch

🧮 Loss Function

Categorical Cross-Entropy Loss (implicitly optimized using softmax gradient)

📈 Results

Final Test Accuracy: ~92% – 94%

Training performed using mini-batch gradient descent

Stable convergence achieved within 20 epochs
# EXPERIMENT 3

Experiment 3: Neural Network for Linear and Non-Linear Classification (From Scratch using NumPy)
📌 Objective

The objective of this experiment is to design and implement a simple neural network from scratch using NumPy to classify two different types of datasets:

Linearly separable dataset

Non-linearly separable dataset

This experiment helps in understanding:

How a perceptron learns a linear decision boundary

Why a single-layer neural network fails for non-linear problems

How hidden layers and non-linear activation functions enable learning complex decision boundaries

📊 Dataset Description

Two synthetic datasets are generated using scikit-learn:

1. Linearly Separable Dataset

Generated using make_classification

Two input features

Clearly separable using a straight line

2. Non-Linearly Separable Dataset

Generated using make_moons

Classes are arranged in curved shapes

Cannot be separated using a linear boundary

🧠 Models Implemented
1️⃣ Single-Layer Perceptron

Used for linearly separable data

Learns a linear decision boundary

Activation: Step function

2️⃣ Multi-Layer Neural Network

Used for non-linearly separable data

One hidden layer

Activation functions:

Sigmoid (hidden layer)

Sigmoid (output layer)

⚙️ Methodology

Generate synthetic datasets

Visualize data distribution

Implement perceptron from scratch

Train perceptron on linear dataset

Evaluate accuracy and decision boundary

Apply perceptron to non-linear data (observe failure)

Implement multi-layer neural network

Train neural network using backpropagation

Visualize non-linear decision boundary

Compare results

📈 Results and Observations
Dataset Type	Model Used	Accuracy	Observation
Linear	Perceptron	~100%	Successfully classified
Non-Linear	Perceptron	~50–60%	Failed to learn
Non-Linear	Multi-Layer NN	~90%+	Successfully classified



# Experiment 6


A deep learning project implementing **Seq2Seq machine translation** from English to Spanish using LSTM-based encoder-decoder architectures with and without attention mechanisms.

---

## 📌 Table of Contents
- [Overview](#overview)
- [Models Implemented](#models-implemented)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Setup & Usage](#setup--usage)
- [Results](#results)
- [Attention Visualizations](#attention-visualizations)
- [Requirements](#requirements)

---

## Overview

This project explores three progressively advanced architectures for neural machine translation:

1. A **vanilla LSTM Encoder-Decoder** that compresses the entire source sentence into a fixed context vector
2. An **LSTM + Bahdanau (Additive) Attention** model that dynamically attends to relevant source words at each decoding step
3. An **LSTM + Luong (Multiplicative) Attention** model using a simpler but effective dot-product alignment score

All models are trained with **teacher forcing** and evaluated using the **BLEU score** metric. Attention weight heatmaps are generated to interpret model behavior.

---

## Models Implemented

### 1. LSTM Encoder-Decoder (No Attention)
- Multi-layer LSTM encoder reads the source sentence and produces a fixed context vector (final hidden + cell states)
- Multi-layer LSTM decoder generates target tokens one at a time conditioned on the context vector
- Teacher forcing applied during training with a 50% ratio

### 2. Bahdanau (Additive) Attention
- At each decoding step, computes alignment scores between the decoder hidden state and **all** encoder outputs
- Score formula: `score = Vᵀ · tanh(W₁·hₜ + W₂·h̄ₛ)`
- Context vector is a weighted sum of encoder outputs, allowing the model to focus on relevant source words

### 3. Luong (Multiplicative) Attention
- Simpler attention using a learned dot product between decoder state and encoder outputs
- Score formula: `score = hₜᵀ · Wₐ · h̄ₛ`
- Computationally lighter while achieving comparable or better results

---

## Dataset

The project uses the **English-Spanish sentence pairs** dataset (tab-separated `.txt` file).

```
Hello.        Hola.
How are you?  ¿Cómo estás?
I am fine.    Estoy bien.
```

| Split | Size |
|-------|------|
| Train | 80%  |
| Validation | 10% |
| Test  | 10%  |

**Recommended sample size:** 50,000 pairs for a good balance of BLEU score and training time.

> 📥 Dataset source: [ManyThings.org Bilingual Sentence Pairs](https://www.manythings.org/anki/) — download `spa-eng.zip`

---

## Project Structure

```
📦 seq2seq-translation/
├── 📓 seq2seq_translation.ipynb   # Main Colab notebook (all code)
├── 📄 README.md                   # This file
├── 📊 loss_comparison.png         # Training/validation loss curves
├── 📊 bleu_comparison.png         # BLEU score bar chart
├── 🖼️  attention_Bahdanau.png     # Bahdanau attention heatmap
├── 🖼️  attention_Luong.png        # Luong attention heatmap
└── 📁 models/                     # Saved model weights (after training)
    ├── model_no_attention.pt
    ├── model_bahdanau.pt
    └── model_luong.pt
```

---

## Setup & Usage

### ▶️ Run in Google Colab (Recommended)

1. Upload `seq2seq_translation.ipynb` to [Google Colab](https://colab.research.google.com/)
2. Upload your dataset (`spa.txt`) to Google Drive
3. Set the runtime to **GPU**: `Runtime > Change runtime type > T4 GPU`
4. In **Section 1** of the notebook, update the dataset path:
   ```python
   DATA_PATH   = '/content/drive/MyDrive/spa.txt'
   MAX_SAMPLES = 50000  # or None for full dataset
   ```
5. Run all cells: `Runtime > Run all`

---

### 💻 Run Locally

```bash
# Clone the repository
git clone https://github.com/your-username/seq2seq-translation.git
cd seq2seq-translation

# Install dependencies
pip install torch sacrebleu nltk scikit-learn matplotlib numpy

# Launch Jupyter
jupyter notebook seq2seq_translation.ipynb
```

---

## Results

> Results below are indicative. Your scores may vary based on dataset size, epochs, and hardware.

| Model | BLEU Score | Val Loss |
|-------|-----------|----------|
| LSTM (No Attention) | ~8–12 | — |
| Bahdanau Attention  | ~18–25 | — |
| Luong Attention     | ~17–24 | — |

**Key Observations:**
- Both attention models significantly outperform the vanilla baseline
- Bahdanau and Luong produce comparable scores; Luong is slightly faster to train
- BLEU improves substantially when trained on 50k+ sentence pairs vs 10k

### Training Curves
![Loss Comparison](loss_comparison.png)

### BLEU Score Comparison
![BLEU Comparison](bleu_comparison.png)

---

## Attention Visualizations

Attention heatmaps show which source (English) words the model focuses on when generating each target (Spanish) word. Brighter cells = higher attention weight.

**Example — Bahdanau Attention:**

![Bahdanau Attention](attention_Bahdanau.png)

**Example — Luong Attention:**

![Luong Attention](attention_Luong.png)

---

## Requirements

| Library | Version |
|---------|---------|
| Python  | 3.8+    |
| PyTorch | 2.0+    |
| sacrebleu | 2.0+ |
| nltk    | 3.8+    |
| scikit-learn | 1.0+ |
| matplotlib | 3.5+ |
| numpy   | 1.21+   |

Install all at once:
```bash
pip install torch sacrebleu nltk scikit-learn matplotlib numpy
```

---

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Embedding Dim | 256 |
| Hidden Dim | 512 |
| LSTM Layers | 2 |
| Dropout | 0.5 |
| Batch Size | 64 |
| Learning Rate | 0.001 |
| Epochs | 20 |
| Teacher Forcing Ratio | 0.5 |
| Optimizer | Adam |

---

## Acknowledgements

- Dataset: [ManyThings.org](https://www.manythings.org/anki/)
- Attention mechanisms based on:
  - [Bahdanau et al., 2015 — "Neural Machine Translation by Jointly Learning to Align and Translate"](https://arxiv.org/abs/1409.0473)
  - [Luong et al., 2015 — "Effective Approaches to Attention-based Neural Machine Translation"](https://arxiv.org/abs/1508.04025)

---

## License

This project is licensed under the MIT License.

