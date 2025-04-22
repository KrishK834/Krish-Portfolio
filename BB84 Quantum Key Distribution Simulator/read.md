# BB84 Quantum Key Distribution Simulator

A simulation of the BB84 quantum key distribution protocol using Qiskit.

## Features

- Alice and Bob exchange qubits using randomly chosen bases.
- Optional eavesdropper (Eve) can intercept and measure qubits.
- Channel noise simulation with configurable error rate.
- Bob measures the qubits and sifts a shared key with Alice.
- Reports error rate, sift rate, and Eve’s information gain.
- Optional visualization of a single qubit exchange.

## Requirements

- Python 3.x
- qiskit
- numpy
- matplotlib

## Usage

```bash
python BB84_QKD_Simulator.py