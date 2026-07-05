# Deep Learning Framework from Scratch

## Overview

This repository contains a small deep learning framework implemented from scratch in Python using NumPy and SciPy. It was developed as a course project at Friedrich-Alexander-University Erlangen-Nuremberg (FAU) to build a practical understanding of neural network internals without relying on high-level deep learning frameworks such as PyTorch or TensorFlow.

The framework includes manually implemented forward and backward passes for common neural network layers, optimizers, losses, and regularization components. It also includes an AlexNet-style convolutional model adapted for MNIST.

## Current Status

- Core neural network layers are implemented from scratch.
- MNIST loading utilities are available.
- `AlexNetMNIST` can be trained on MNIST.
- The training script supports a train/validation split.
- Training reports mini-batch train loss and accuracy.
- Validation loss and accuracy can be reported periodically.
- Final test loss and accuracy can be computed after training.

## Repository Structure

```text
Layers/          Core neural network layers and data helpers
Models/          Model definitions, including AlexNet and AlexNetMNIST
Optimization/    Loss functions, optimizers, and constraints
scripts/         Training, smoke test, gradient check, and data download scripts
Data/            Local MNIST dataset files
NeuralNetwork.py Base neural network container and training/test methods
```

## Features

- Fully connected layers
- Convolution layers
- Max pooling
- Flatten
- Dropout
- Batch normalization
- ReLU, Sigmoid, Tanh, and SoftMax activations
- Cross-entropy loss
- SGD
- SGD with momentum
- Adam optimizer
- RNN and LSTM layers
- MNIST data loading
- Gradient checking utilities

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If MNIST is not already available locally, download it:

```bash
python scripts/download_mnist.py
```

## Training AlexNetMNIST

Run:

```bash
python scripts/train_alexnet_mnist.py
```

The training script:

- loads MNIST data
- splits the original training set into training and validation subsets
- trains `AlexNetMNIST`
- prints mini-batch training loss and accuracy
- periodically prints validation loss and accuracy
- evaluates test loss and accuracy after training

The model expects MNIST images in this shape:

```text
(batch_size, 1, 28, 28)
```

and produces class probabilities in this shape:

```text
(batch_size, 10)
```

## AlexNetMNIST Architecture

`AlexNetMNIST` is a smaller AlexNet-style convolutional network adapted for 28x28 grayscale MNIST images:

```text
Conv(1 -> 32, 5x5)
ReLU
MaxPool(2x2)
Conv(32 -> 64, 5x5)
ReLU
MaxPool(2x2)
Flatten
FullyConnected(64 * 7 * 7 -> 256)
ReLU
Dropout(0.5)
FullyConnected(256 -> 10)
SoftMax
```

## Limitations

- This is a CPU-only NumPy/SciPy implementation.
- Training is much slower than using PyTorch, TensorFlow, JAX, or another optimized framework.
- The implementation prioritizes learning and transparency over performance.
- Gradients are mostly accumulated as summed batch gradients, so learning-rate behavior depends on batch size.
- No model checkpointing or weight serialization is currently implemented.
- GPU acceleration through CUDA, MPS, or CuPy is not currently implemented.

## Future Work

Potential improvements:

- Switch to a mean-loss and averaged-gradient convention.
- Add model saving and loading.
- Add cleaner metric logging.
- Evaluate full validation and test sets efficiently.
- Add benchmarks for core operations.
- Experiment with CUDA or CuPy acceleration for selected operators.
## License & Attribution

The neural-network layers, optimizers, and models in this repository are my own
implementations, written for the Deep Learning course at Friedrich-Alexander-Universität
Erlangen-Nürnberg (FAU). This original work is released under the [MIT License](LICENSE).

One file, `Layers/Helpers.py` (gradient checking, dataset loaders, and plotting
utilities), was **provided by the course** and is not my own work; it retains its
original attribution in the file header. It is included only so the framework runs
end to end, and is **not** covered by the MIT License — its rights remain with the
original authors.
