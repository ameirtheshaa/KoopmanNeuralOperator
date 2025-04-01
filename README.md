# DynamicalOperator

A deep learning framework for learning dynamical systems using Koopman operator theory and neural networks.

## Overview

DynamicalOperator is a PyTorch-based implementation of a neural operator framework that combines spectral methods with deep learning to learn and predict the evolution of dynamical systems. The framework is designed to effectively handle both 1D and 2D dynamical systems by leveraging the power of Koopman operator theory.

The key insight of this approach is to transform complex nonlinear dynamics into a linear representation in a higher-dimensional space, where prediction becomes more tractable.

## Features

- **Spectral Methods**: Efficient computation in Fourier space for modeling dynamical systems
- **Neural Operators**: Combines neural networks with operator learning
- **Encoder-Decoder Architecture**: Transforms data to and from a latent space where the dynamics are more easily modeled
- **1D and 2D Support**: Handles both one-dimensional and two-dimensional spatial domains
- **Autoregressive Prediction**: Capable of long-term forecasting through autoregressive prediction

## Installation

```bash
git clone https://github.com/yourusername/dynamical-operator.git
cd dynamical-operator
pip install -r requirements.txt
```

## Usage

The basic usage pattern follows these steps:

1. Prepare your time-series data
2. Initialize the DynamicalOperator with appropriate parameters
3. Train the model
4. Generate predictions

### Example with Lorenz System

```python
import torch
from dynamical_operator import DynamicalOperator

# Assuming 'data' is a tensor with shape [num_samples, features, height, time]
# where height can be 1 for 1D systems

# Initialize model
model = DynamicalOperator(
    training_data=data,
    time_horizon=50,             # Input sequence length
    latent_dim=32,               # Latent dimension size
    fourier_modes=16,            # Number of Fourier modes to use
    iterations=8,                # Number of operator iterations
    device='cuda',               # 'cuda' or 'cpu'
    architecture='DNO1d',        # 'DNO1d' or 'DNO2d'
    batch_size=32,               # Batch size for training
    epochs=50                    # Number of training epochs
)

# Generate predictions
predictions = model.generate_predictions()
```

For a complete working example, see the provided `lorenz_test.py` script.

## Mathematical Background

This implementation is based on Koopman operator theory, which provides a way to transform nonlinear dynamical systems into linear operators in a higher-dimensional space. The key steps are:

1. **Encoding**: Transform the input data to a higher-dimensional latent space
2. **Spectral Evolution**: Apply a learned operator in Fourier space to evolve the system
3. **Decoding**: Transform the evolved state back to the original space

By learning this process from data, the model can capture complex dynamical behaviors and make accurate predictions.

## Citation

If you use this code in your research, please cite:

```
@misc{DynamicalOperator2025,
  author = {Ameir Shaa, Claude Guet},
  title = {Koopman Nueral Operator: Neural Operators for Learning Dynamical Systems in context of Fusion Sciences},
  <!-- year = {2025}, -->
  <!-- publisher = {GitHub}, -->
  <!-- journal = {PRX}, -->
  <!-- howpublished = {\url{https://github.com/yourusername/dynamical-operator}} -->
}
```

## License

MIT

## Acknowledgements

This project draws inspiration from developments in neural operators, Koopman theory, and deep learning for dynamical systems.