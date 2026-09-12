# Character Recognition Neural Network from Scratch

A fully-connected neural network built **using only NumPy** to classify
**35 classes**: uppercase letters **A–Z** and digits **1–9**.

No TensorFlow, PyTorch, or Keras — every gradient is derived analytically.

---

## Features

- Pure NumPy implementation (forward, backward, gradient descent)
- Configurable activation: **ReLU** or **Tanh**
- Synthetic dataset: **50+ samples per class** via PIL
- Stratified **train / val / test** split
- Softmax output + cross-entropy loss + L2 regularization
- Confusion matrix, training curves, error analysis

---

## Requirements

```bash
pip install numpy matplotlib scikit-learn seaborn Pillow

```

##architecture
Input (784) → Linear(128) → ReLU → Linear(64) → ReLU → Linear(35) → Softmax

##training loop
FOR each epoch:
    Shuffle training data
    FOR each mini-batch:
        1. Forward propagation
        2. Cross-entropy loss (+ L2)
        3. Backpropagation
        4. Gradient descent update
    Validation: forward pass on val set
