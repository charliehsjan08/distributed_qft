import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

def create_unprotected_qubit(drift_angle):
    """Simulates a qubit waiting in a queue, suffering from phase drift."""
    circ = QuantumCircuit(1, 1)
    
    # 1. Prepare sensitive superposition state |+>
    circ.h(0)
    circ.barrier()
    
    # 2. THE QUEUE (Idle Time)
    # The qubit sits idle and accumulates a stray phase rotation (Dephasing)
    circ.rz(drift_angle, 0)
    circ.barrier()
    
    # 3. Measure back in X-basis
    circ.h(0)
    circ.measure(0, 0)
    return circ

def create_protected_qubit(drift_angle):
    """Simulates a qubit waiting in a queue, protected by a Spin Echo."""
    circ = QuantumCircuit(1, 1)
    
    # 1. Prepare sensitive superposition state |+>
    circ.h(0)
    circ.barrier()
    
    # 2. THE QUEUE WITH SPIN ECHO (Dynamical Decoupling)
    # We split the idle time in half, and insert X gates to flip the state.
    
    # First half of the queue wait...
    circ.rz(drift_angle / 2, 0)
    
    # The Spin Echo (Flip the qubit upside down)
    circ.x(0)
    
    # Second half of the queue wait... 
    # Because the qubit is flipped, the stray magnetic field now rewinds the error!
    circ.rz(drift_angle / 2, 0)
    
    # Flip it back to normal
    circ.x(0)
    circ.barrier()
    
    # 3. Measure back in X-basis
    circ.h(0)
    circ.measure(0, 0)
    return circ

def run_spin_echo_test():
    print("--- Testing Spin Echo Logic for Distributed Queues ---")
    
    # Let's say the stray magnetic field rotates the phase by PI/2 during the wait
    drift_angle = np.pi / 2 
    
    unprotected_circ = create_unprotected_qubit(drift_angle)
    protected_circ = create_protected_qubit(drift_angle)
    
    simulator = AerSimulator()
    
    # Run Unprotected
    unprotected_counts = simulator.run(unprotected_circ, shots=10000).result().get_counts()
    unprotected_fid = unprotected_counts.get('0', 0) / 10000
    
    # Run Protected
    protected_counts = simulator.run(protected_circ, shots=10000).result().get_counts()
    protected_fid = protected_counts.get('0', 0) / 10000
    
    print(f"Unprotected Qubit Fidelity (Destroyed by Dephasing): {unprotected_fid * 100:.2f}%")
    print(f"Protected Qubit Fidelity (Saved by Spin Echo):       {protected_fid * 100:.2f}%")

if __name__ == "__main__":
    run_spin_echo_test()