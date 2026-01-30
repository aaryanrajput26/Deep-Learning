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
