import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFT
from src.circuits import create_registers, build_state_preparation, build_distributed_qft
from src.hardware_noise import get_comprehensive_noise_model
from src.execution import run_local_simulation
from src.readout_mitigation import apply_matrix_inversion
from qiskit.quantum_info import Operator
from qiskit.circuit.library import UnitaryGate

def generate_base_circuit(approx_degree):
    """
    Generates the QFT architecture with an APPROXIMATE forward path 
    and a mathematically PERFECT, NOISELESS inverse path.
    """
    qr_data_A, qr_comm_A, qr_comm_B, qr_data_B, cr_tele, cr_out = create_registers()
    data_qubits = [qr_data_A[i] if i < 4 else qr_data_B[i-4] for i in range(8)]
    V_prep = build_state_preparation(data_qubits)
    
    circ = QuantumCircuit(qr_data_A, qr_comm_A, qr_comm_B, qr_data_B, cr_tele, cr_out)
    circ.compose(V_prep, qubits=data_qubits, inplace=True)
    circ.barrier()
    
    # 1. Forward Path (Noisy, Approximate, Hardware-Dependent)
    _ = build_distributed_qft(circ, (qr_data_A, qr_comm_A, qr_comm_B, qr_data_B, cr_tele, cr_out), approx_degree=approx_degree)
    circ.barrier()
    
    # 2. Conceptual Exact Inverse (Not added to circuit yet)
    qft_exact = QFT(num_qubits=8, approximation_degree=0, do_swaps=False)
    ideal_inverse = QuantumCircuit(8)
    ideal_inverse.compose(qft_exact.inverse(), inplace=True)
    ideal_inverse.compose(V_prep.inverse(), inplace=True)
    
    # 3. The Noiseless Bypass
    # Convert the inverse to a pure matrix, and wrap it in a UnitaryGate
    inv_matrix = Operator(ideal_inverse).data
    inv_gate = UnitaryGate(inv_matrix, label="Ideal_Uncompute")
    
    # Append the pure math gate. Aer will simulate this instantly with 0% noise.
    circ.append(inv_gate, data_qubits)
    circ.barrier()
    
    # 4. Final Measurement (Subject to SPAM, fixed by M3)
    circ.measure(data_qubits, cr_out)
    
    return circ

def execute_aqft_sweep():
    print("--- Executing AQFT Optimization Sweep ---")
    num_shots = 5000 # Lowered slightly for faster sweep times
    degrees = list(range(1,9)) # Sweeping from 0 to 6
    
    base_err_1q = 0.0005
    base_err_2q = 0.002
    base_spam = 0.01
    
    final_fidelities = []
    
    for degree in degrees:
        print(f"\n--- Testing Approximation Degree: {degree} ---")
        circ = generate_base_circuit(degree)
        m3_fidelities = []
        
        # Run the ZNE scales for this specific degree
        for scale in [1, 3, 5]:
            scaled_model = get_comprehensive_noise_model(
                t1_us=np.inf, t2_us=np.inf, 
                pulse_err_1q=base_err_1q * scale, 
                pulse_err_2q=base_err_2q * scale, 
                spam_error=base_spam * scale
            )
            
            counts = run_local_simulation(circ, noise_model=scaled_model, shots=num_shots)
            m3_fid = apply_matrix_inversion(counts, num_shots, p_error=base_spam * scale)
            m3_fidelities.append(m3_fid)
            
        # ZNE Math
        f1, f3, f5 = m3_fidelities
        mitigated_fid = (15 * f1 - 10 * f3 + 3 * f5) / 8
        final_mitigated = min(max(mitigated_fid, 0), 1.0)
        
        final_fidelities.append(final_mitigated)
        print(f"Resulting Sweet Spot Fidelity: {final_mitigated * 100:.2f}%")

    # Generate the Optimization Plot
    plt.figure(figsize=(10, 6))
    plt.plot(degrees, [f * 100 for f in final_fidelities], marker='D', linestyle='-', color='purple', linewidth=2)
    
    plt.title('AQFT Optimization: Finding the Sweet Spot')
    plt.xlabel('Approximation Degree (Dropped Rotation Layers)')
    plt.ylabel('Fully Mitigated Output Fidelity (%)')
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    
    plt.savefig('aqft_sweet_spot.png', dpi=300)
    print("\nSweep Complete! Check 'aqft_sweet_spot.png' to find your optimal parameter.")

if __name__ == "__main__":
    execute_aqft_sweep()