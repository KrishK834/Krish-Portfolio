import numpy as np
import random
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_bloch_multivector

class BB84Simulation:
    def __init__(self, key_length=100, eavesdropper=False, error_rate=0):
        
        self.key_length = key_length
        self.eavesdropper = eavesdropper
        self.error_rate = error_rate
        
        self.simulator = AerSimulator()
        
        self.alice_bits = []    
        self.alice_bases = []   
        self.bob_bases = []     
        self.bob_results = []   
        self.shared_key = []    
        
        self.eve_bases = []    
        self.eve_results = []   
        
    def alice_prepares_qubits(self):
        self.alice_bits = [random.randint(0, 1) for _ in range(self.key_length)]
        self.alice_bases = [random.randint(0, 1) for _ in range(self.key_length)]
        
        
        circuits = []
        for bit, basis in zip(self.alice_bits, self.alice_bases):
            qc = QuantumCircuit(1, 1)
            
            if bit:
                qc.x(0)
            
            if basis:
                qc.h(0)
                
            circuits.append(qc)
        
        return circuits
    
    def eve_intercepts(self, circuits):
        if not self.eavesdropper:
            return circuits
            
        self.eve_bases = [random.randint(0, 1) for _ in range(self.key_length)]
        self.eve_results = []
        
        eve_circuits = []
        
        for i, qc in enumerate(circuits):
            eve_qc = qc.copy()
            
            if self.eve_bases[i] == 1:  
                eve_qc.h(0)
            
            statevector_qc = eve_qc.copy()
            statevector_qc.save_statevector()
            
            transpiled = transpile(statevector_qc, self.simulator)
            job = self.simulator.run(transpiled)
            state = job.result().get_statevector()
            
            prob_0 = abs(state[0])**2
            result = 0 if random.random() < prob_0 else 1
            self.eve_results.append(result)
            
            new_qc = QuantumCircuit(1, 1)
            if result:
                new_qc.x(0)
            
            if self.eve_bases[i]:
                new_qc.h(0)
            
            if self.eve_bases[i] != self.alice_bases[i]:
                
                if self.alice_bases[i]:
                    new_qc.h(0)  
                else:
                    new_qc.h(0)  
            
            eve_circuits.append(new_qc)
            
        return eve_circuits
    
    def apply_channel_noise(self, circuits):
        if self.error_rate <= 0:
            return circuits
            
        noisy_circuits = []
        for qc in circuits:
            noisy_qc = qc.copy()
            
            if random.random() < self.error_rate:
                noisy_qc.x(0)  
            noisy_circuits.append(noisy_qc)
        
        return noisy_circuits
    
    def bob_measures_qubits(self, circuits):
        self.bob_bases = [random.randint(0, 1) for _ in range(self.key_length)]
        self.bob_results = []
        
        for i, qc in enumerate(circuits):
            
            bob_qc = qc.copy()
            
            if self.bob_bases[i]:
                bob_qc.h(0)
                
            bob_qc.measure(0, 0)
            
            transpiled_circuit = transpile(bob_qc, self.simulator)
            result = self.simulator.run(transpiled_circuit, shots=1).result()
            counts = result.get_counts()
            measured_bit = int(list(counts.keys())[0])
            self.bob_results.append(measured_bit)
    
    def sift_key(self):
        self.shared_key = []
        matching_indices = []
        
        for i in range(self.key_length):
            if self.alice_bases[i] == self.bob_bases[i]:
                matching_indices.append(i)
                self.shared_key.append(self.alice_bits[i])
                
        
        errors = sum(self.alice_bits[i] != self.bob_results[i] 
                     for i in matching_indices)
        error_rate = errors / len(matching_indices) if matching_indices else 0
        
        return {
            'matching_indices': matching_indices,
            'raw_key_length': len(matching_indices),
            'error_rate': error_rate,
            'errors': errors
        }
    
    def run_simulation(self):
        
        circuits = self.alice_prepares_qubits()
        
        if self.eavesdropper:
            circuits = self.eve_intercepts(circuits)
        
        circuits = self.apply_channel_noise(circuits)
        
        self.bob_measures_qubits(circuits)
        
        stats = self.sift_key()
        
        return stats
    
    def get_simulation_results(self):
        return {
            'alice_bits': self.alice_bits,
            'alice_bases': self.alice_bases,
            'bob_bases': self.bob_bases,
            'bob_results': self.bob_results,
            'shared_key': self.shared_key,
            'eve_bases': self.eve_bases if self.eavesdropper else None,
            'eve_results': self.eve_results if self.eavesdropper else None,
            'eavesdropper_present': self.eavesdropper,
            'error_rate': self.error_rate
        }
    
    def visualize_single_qubit_exchange(self, qubit_index=0):
        if qubit_index >= self.key_length:
            raise ValueError(f"Qubit index {qubit_index} out of range")
        
        alice_qc = QuantumCircuit(1, 1)
        if self.alice_bits[qubit_index]:
            alice_qc.x(0)
        if self.alice_bases[qubit_index]:
            alice_qc.h(0)
        
        statevector_qc = alice_qc.copy()
        statevector_qc.save_statevector()
        transpiled = transpile(statevector_qc, self.simulator)
        job = self.simulator.run(transpiled)
        alice_state = job.result().get_statevector()
        
        bob_basis = "X" if self.bob_bases[qubit_index] else "Z"
        alice_basis = "X" if self.alice_bases[qubit_index] else "Z"
        
        bloch_fig = plot_bloch_multivector(alice_state)
        plt.title(f"Alice's Qubit (Bit: {self.alice_bits[qubit_index]}, Basis: {alice_basis})")
        
        info_fig, ax = plt.subplots(figsize=(6, 5))
        ax.axis('off')
        ax.text(0.5, 0.7, f"Bob measures in {bob_basis} basis", 
              horizontalalignment='center', fontsize=14)
        ax.text(0.5, 0.5, f"Bob's result: {self.bob_results[qubit_index]}", 
              horizontalalignment='center', fontsize=14)
        ax.text(0.5, 0.3, f"Bases match: {self.alice_bases[qubit_index] == self.bob_bases[qubit_index]}", 
              horizontalalignment='center', fontsize=14)
        
        plt.tight_layout()
        
        return bloch_fig, info_fig

def run_interactive_simulation():
    print("=====================================================")
    print("   BB84 Quantum Key Distribution Protocol Simulator  ")
    print("=====================================================")
    
    try:
        key_length = int(input("Enter desired key length (default: 100): ") or 100)
        
        eve_choice = input("Include eavesdropper? (y/n, default: n): ").lower()
        eavesdropper = eve_choice == 'y'
        
        noise_choice = input("Include channel noise? (y/n, default: n): ").lower()
        if noise_choice == 'y':
            error_rate = float(input("Enter noise error rate (0.0-1.0, default: 0.05): ") or 0.05)
        else:
            error_rate = 0.0
            
        print("\nInitializing simulation...")
        sim = BB84Simulation(key_length=key_length, eavesdropper=eavesdropper, error_rate=error_rate)
        
        print("Running BB84 protocol...")
        stats = sim.run_simulation()
        
        print("\n=== Simulation Results ===")
        print(f"Raw key length: {stats['raw_key_length']} bits")
        print(f"Error rate: {stats['error_rate'] * 100:.2f}%")
        print(f"Errors detected: {stats['errors']}")
        print(f"First 20 bits of sifted key: {sim.shared_key[:20]}")
        
        sift_rate = stats['raw_key_length'] / key_length
        print(f"\nSift rate (key bits / qubits): {sift_rate:.2f}")
        print(f"Effective key rate: {stats['raw_key_length'] / key_length:.2f} bits per qubit sent")
        
        if eavesdropper:
            print("\n=== Eavesdropper Analysis ===")
            eve_correct = sum(sim.eve_results[i] == sim.alice_bits[i] for i in range(key_length)
                            if sim.eve_bases[i] == sim.alice_bases[i])
            eve_total = sum(1 for i in range(key_length) if sim.eve_bases[i] == sim.alice_bases[i])
            eve_accuracy = eve_correct / eve_total if eve_total > 0 else 0
            print(f"Eve's information gain: {eve_accuracy:.2f} of intercepted qubits")
            
        if input("\nVisualize a single qubit exchange? (y/n): ").lower() == 'y':
            try:
                qubit_idx = min(0, key_length-1)  
                
                figures = sim.visualize_single_qubit_exchange(qubit_idx)
                plt.show()
            except Exception as e:
                print(f"Visualization error: {e}")
                print("Make sure you're in an environment that supports matplotlib display.")
                
    except Exception as e:
        print(f"Error during simulation: {e}")
        print("Please check your inputs and try again.")

if __name__ == "__main__":
    run_interactive_simulation()