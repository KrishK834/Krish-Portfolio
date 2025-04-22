# Quantum Algorithms Demonstrator

A simulation of core quantum computing algorithms using Qiskit.

## Features

- **Quantum Teleportation**  
  Simulates teleportation of an arbitrary quantum state using entanglement and classical communication.  
  Displays measurement outcomes and interprets teleportation success.

- **Quantum Fourier Transform (QFT)**  
  Performs the QFT on a 3-qubit register.  
  Shows intermediate steps like controlled phase rotations and qubit swaps.  
  Outputs measurement probabilities in the frequency basis.

- **Grover's Search Algorithm**  
  Implements Grover’s search for a marked item in an unstructured database.  
  Accepts custom target states (e.g., `|101⟩`).  
  Automatically calculates number of Grover iterations.  
  Displays probability of successful detection.

## Requirements

- Python 3.x  
- qiskit  
- qiskit-aer  
- numpy  
- matplotlib

## Usage

```bash
python quantum_algorithm_simulator.py