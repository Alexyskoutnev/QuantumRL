# Quantum Reinforcement Learning

This repository contains Python code for implementing Quantum Reinforcement Learning (QRL) using Variational Quantum Circuits (VQC). QRL leverages the principles of quantum computing to solve reinforcement learning tasks.

## Environment

The `QuantumGridWorld` environment is used for training and evaluation. This environment represents a grid world where an agent navigates through states to achieve a goal. 

## Dependencies

- `pennylane`: A Python library for quantum machine learning.
- `torch`: PyTorch, a popular machine learning library.
- `gymnasium`: A custom gym environment for the QuantumGridWorld.
- Other standard Python libraries like `numpy`, `random`, `pickle`, `matplotlib`, and `collections`.

## Components

### Quantum Reinforcement Learning Algorithm

The main algorithm consists of training a variational quantum circuit to approximate the Q-values of state-action pairs. The circuit is trained using a Q-learning approach with experience replay.

### Quantum Circuit

The variational quantum circuit is implemented using PennyLane. It consists of multiple layers, each containing rotations and entangling gates.

### IBM API Token

To connect to the IBM Quantum Computing Server, you will need to create a `config.toml` file with the following structure:


```
[qiskit.global]

  [qiskit.ibmq]
  ibmqx_token = "YOUR API KEY HERE"
```